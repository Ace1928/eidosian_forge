import random
import threading
from oslo_utils import reflection
from oslo_utils import timeutils
from taskflow.engines.worker_based import protocol as pr
from taskflow import logging
from taskflow.utils import kombu_utils as ku
@property
def messages_processed(self):
    """How many notify response messages have been processed."""
    return self._messages_processed