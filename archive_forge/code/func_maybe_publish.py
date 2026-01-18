import random
import threading
from oslo_utils import reflection
from oslo_utils import timeutils
from taskflow.engines.worker_based import protocol as pr
from taskflow import logging
from taskflow.utils import kombu_utils as ku
def maybe_publish(self):
    """Periodically called to publish notify message to each topic.

        These messages (especially the responses) are how this find learns
        about workers and what tasks they can perform (so that we can then
        match workers to tasks to run).
        """
    if self._messages_published == 0:
        self._proxy.publish(pr.Notify(), self._topics, reply_to=self._uuid)
        self._messages_published += 1
        self._watch.restart()
    elif self._watch.expired():
        self._proxy.publish(pr.Notify(), self._topics, reply_to=self._uuid)
        self._messages_published += 1
        self._watch.restart()