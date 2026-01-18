from tensorflow.core.protobuf import cluster_pb2
from tensorflow.python.distribute import distribute_coordinator_context as dc_context
from tensorflow.python.training import server_lib
def should_load_checkpoint():
    """Returns whether the current worker should load checkpoints.

  In multi-worker training, if loading checkpoint is requested by user, or
  needed for fault-tolerance, the cluster should load checkpoint but not
  necessarily every worker in the cluster should.

  Returns:
      Whether this particular worker in the cluster should load checkpoints.
  """
    return dc_context.get_current_worker_context().experimental_should_init