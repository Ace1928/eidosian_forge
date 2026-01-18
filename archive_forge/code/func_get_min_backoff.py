import os
import ovs.util
import ovs.vlog
def get_min_backoff(self):
    """Return the minimum number of milliseconds to back off between
        consecutive connection attempts.  The default is 1000 ms."""
    return self.min_backoff