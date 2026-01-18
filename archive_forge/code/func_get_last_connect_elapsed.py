import os
import ovs.util
import ovs.vlog
def get_last_connect_elapsed(self, now):
    """Returns the number of milliseconds since 'fsm' was last connected
        to its peer. Returns None if never connected."""
    if self.last_connected:
        return now - self.last_connected
    else:
        return None