from mlflow.environment_variables import (
from mlflow.utils.annotations import experimental
@experimental
def set_system_metrics_node_id(node_id):
    """Set the system metrics node id.

    node_id is the identifier of the machine where the metrics are collected. This is useful in
    multi-node (distributed training) setup.
    """
    if node_id is None:
        MLFLOW_SYSTEM_METRICS_NODE_ID.unset()
    else:
        MLFLOW_SYSTEM_METRICS_NODE_ID.set(node_id)