import logging
import queue
import threading
from oslo_utils import eventletutils
from oslo_utils import importutils
def get_executor_with_context():
    if eventletutils.is_monkey_patched('thread'):
        LOG.debug('Threading is patched, using an eventlet executor')
        return 'eventlet'
    LOG.debug('Using a threading executor')
    return 'threading'