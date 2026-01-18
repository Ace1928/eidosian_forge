import asyncio
import logging
import time
import traceback
from .compatibility import guarantee_single_callable
def get_or_create_application_instance(self, scope_id, scope):
    """
        Creates an application instance and returns its queue.
        """
    if scope_id in self.application_instances:
        self.application_instances[scope_id]['last_used'] = time.time()
        return self.application_instances[scope_id]['input_queue']
    while len(self.application_instances) > self.max_applications:
        self.delete_oldest_application_instance()
    input_queue = asyncio.Queue()
    application_instance = guarantee_single_callable(self.application)
    future = asyncio.ensure_future(application_instance(scope=scope, receive=input_queue.get, send=lambda message: self.application_send(scope, message)))
    self.application_instances[scope_id] = {'input_queue': input_queue, 'future': future, 'scope': scope, 'last_used': time.time()}
    return input_queue