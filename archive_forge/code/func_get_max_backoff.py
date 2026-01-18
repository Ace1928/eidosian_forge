import os
import ovs.util
import ovs.vlog
def get_max_backoff(self):
    """Return the maximum number of milliseconds to back off between
        consecutive connection attempts.  The default is 8000 ms."""
    return self.max_backoff