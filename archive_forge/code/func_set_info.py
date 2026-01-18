import ray
import logging
def set_info(self, ids, world_size, rank, backend):
    """Store collective information."""
    self.ids = ids
    self.world_size = world_size
    self.rank = rank
    self.backend = backend