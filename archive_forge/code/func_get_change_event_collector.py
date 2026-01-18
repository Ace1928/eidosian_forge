import inspect
import os
import threading
from contextlib import contextmanager
from datetime import datetime
from traits import trait_notifiers
def get_change_event_collector(self, thread_name):
    """ Return the dedicated RecordContainer for the thread.

        If no RecordContainer is found for `thread_name` then a new
        RecordContainer is created.

        """
    with self._creation_lock:
        container = self._record_containers.get(thread_name)
        if container is None:
            container = RecordContainer()
            self._record_containers[thread_name] = container
        return container