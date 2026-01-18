from tensorflow.core.protobuf import cluster_pb2
from tensorflow.python.distribute import distribute_coordinator_context as dc_context
from tensorflow.python.training import server_lib
def coordination_leader(cluster_spec):
    """Return the task name of the coordination service leader.

  Args:
    cluster_spec: a dict, `ClusterDef` or `ClusterSpec` object sxpecifying the
      cluster configurations.

  Returns:
    a string indicating the task name of the coordination service leader.
  """
    cluster_spec = normalize_cluster_spec(cluster_spec)
    if not cluster_spec.as_dict():
        return ''
    if 'ps' in cluster_spec.jobs:
        return '/job:ps/replica:0/task:0'
    if 'chief' in cluster_spec.jobs:
        return '/job:chief/replica:0/task:0'
    assert 'worker' in cluster_spec.jobs
    return '/job:worker/replica:0/task:0'