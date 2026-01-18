import threading
def is_stopped(self):
    """Returns if the timeout has been interrupted."""
    return self._event.is_set()