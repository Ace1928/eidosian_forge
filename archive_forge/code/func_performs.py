import random
import threading
from oslo_utils import reflection
from oslo_utils import timeutils
from taskflow.engines.worker_based import protocol as pr
from taskflow import logging
from taskflow.utils import kombu_utils as ku
def performs(self, task):
    if not isinstance(task, str):
        task = reflection.get_class_name(task)
    return task in self.tasks