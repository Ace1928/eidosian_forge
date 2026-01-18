from tensorflow.core.protobuf import cluster_pb2
from tensorflow.python.distribute import distribute_coordinator_context as dc_context
from tensorflow.python.training import server_lib
def task_count(cluster_spec, task_type):
    try:
        return cluster_spec.num_tasks(task_type)
    except ValueError:
        return 0