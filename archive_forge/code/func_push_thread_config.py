import re
import threading
def push_thread_config(self, conf):
    """
        Make ``conf`` the active configuration for this thread.
        Thread-local configuration always overrides process-wide
        configuration.

        This should be used like::

            conf = make_conf()
            dispatching_config.push_thread_config(conf)
            try:
                ... do stuff ...
            finally:
                dispatching_config.pop_thread_config(conf)
        """
    local_dict().setdefault(self._local_key, []).append(conf)