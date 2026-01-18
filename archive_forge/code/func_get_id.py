import ray
import logging
def get_id(self):
    """Get the NCCL unique ID held in this store."""
    if not self.nccl_id:
        logger.warning('The NCCL ID has not been set yet for store {}.'.format(self.name))
    return self.nccl_id