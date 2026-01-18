import os
import ovs.util
import ovs.vlog
def get_last_disconnect_elapsed(self, now):
    """Returns the number of milliseconds since 'fsm' was last disconnected
        from its peer. Returns None if never disconnected."""
    if self.last_disconnected:
        return now - self.last_disconnected
    else:
        return None