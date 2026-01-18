import ray
import logging
def set_id(self, uid):
    """
        Initialize the NCCL unique ID for this store.

        Args:
            uid: the unique ID generated via the NCCL get_unique_id API.

        Returns:
            None
        """
    self.nccl_id = uid
    return self.nccl_id