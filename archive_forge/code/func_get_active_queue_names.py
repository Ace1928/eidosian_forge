import re
import inspect
import traceback
import copy
import logging
import hmac
from base64 import b64decode
import tornado
from ..utils import template, bugreport, strtobool
def get_active_queue_names(self):
    queues = set([])
    for _, info in self.application.workers.items():
        for queue in info.get('active_queues', []):
            queues.add(queue['name'])
    if not queues:
        queues = set([self.capp.conf.task_default_queue]) | {q.name for q in self.capp.conf.task_queues or [] if q.name}
    return sorted(queues)