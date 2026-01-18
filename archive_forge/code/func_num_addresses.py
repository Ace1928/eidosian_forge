import functools
@property
def num_addresses(self):
    """Number of hosts in the current subnet."""
    return int(self.broadcast_address) - int(self.network_address) + 1