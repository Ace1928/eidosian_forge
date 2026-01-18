import os
import ovs.util
import ovs.vlog
def set_backoff_free_tries(self, backoff_free_tries):
    """Sets the number of connection attempts that will be made without
        backoff to 'backoff_free_tries'.  Values 0 and 1 both
        represent a single attempt."""
    self.backoff_free_tries = backoff_free_tries