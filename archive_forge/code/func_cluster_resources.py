import json
import logging
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Union
from ray._private import ray_option_utils
from ray.util.client.runtime_context import _ClientWorkerPropertyAPI
def cluster_resources(self):
    """Get the current total cluster resources.

        Note that this information can grow stale as nodes are added to or
        removed from the cluster.

        Returns:
            A dictionary mapping resource name to the total quantity of that
                resource in the cluster.
        """
    import ray.core.generated.ray_client_pb2 as ray_client_pb2
    return self.worker.get_cluster_info(ray_client_pb2.ClusterInfoType.CLUSTER_RESOURCES)