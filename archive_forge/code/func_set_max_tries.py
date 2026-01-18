import os
import ovs.util
import ovs.vlog
def set_max_tries(self, max_tries):
    """Limits the maximum number of times that this object will ask the
        client to try to reconnect to 'max_tries'.  None (the default) means an
        unlimited number of tries.

        After the number of tries has expired, the FSM will disable itself
        instead of backing off and retrying."""
    self.max_tries = max_tries