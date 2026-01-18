from rpy2.rinterface import process_revents
import time
import warnings
import threading
def start(self):
    """ start the event processor """
    if self._thread is not None and self._thread.is_alive():
        raise warnings.warn('Processing of R events already started.')
    else:
        self._thread = _EventProcessorThread(name=self.name_thread)
        self._thread.setDaemon(self.daemon_thread)
        self._thread.start()