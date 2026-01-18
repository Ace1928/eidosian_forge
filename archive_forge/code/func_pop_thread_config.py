import re
import threading
def pop_thread_config(self, conf=None):
    """
        Remove a thread-local configuration.  If ``conf`` is given,
        it is checked against the popped configuration and an error
        is emitted if they don't match.
        """
    self._pop_from(local_dict()[self._local_key], conf)