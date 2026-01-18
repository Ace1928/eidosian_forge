import asyncio
import logging
import time
import traceback
from .compatibility import guarantee_single_callable
def delete_application_instance(self, scope_id):
    """
        Removes an application instance (makes sure its task is stopped,
        then removes it from the current set)
        """
    details = self.application_instances[scope_id]
    del self.application_instances[scope_id]
    if not details['future'].done():
        details['future'].cancel()