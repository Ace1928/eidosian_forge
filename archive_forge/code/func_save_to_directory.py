import inspect
import os
import threading
from contextlib import contextmanager
from datetime import datetime
from traits import trait_notifiers
def save_to_directory(self, directory_name):
    """ Save records files into the directory.

        Each RecordContainer will dump its records on a separate file named
        <thread_name>.trace.

        """
    with self._creation_lock:
        containers = self._record_containers
        for thread_name, container in containers.items():
            filename = os.path.join(directory_name, '{0}.trace'.format(thread_name))
            container.save_to_file(filename)