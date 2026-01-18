import numpy
from ray.util.collective.types import ReduceOp, torch_available
def create_nccl_communicator(world_size, nccl_unique_id, rank):
    """Create an NCCL communicator using NCCL APIs.

    Args:
        world_size: the number of processes of this communicator group.
        nccl_unique_id: the NCCLUniqueID for this group.
        rank: the rank of this process.
    Returns:
        comm (nccl.ncclComm_t): an NCCL communicator.
    """
    comm = NcclCommunicator(world_size, nccl_unique_id, rank)
    return comm