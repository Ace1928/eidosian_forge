import ray
import logging
def get_info(self):
    """Get previously stored collective information."""
    return (self.ids, self.world_size, self.rank, self.backend)