import time
from typing import Any, Callable, Iterable, List, Tuple, Union
import ray
from ray import ObjectRef
from ray.cluster_utils import Cluster
def build_cluster(num_nodes, num_cpus, object_store_memory):
    cluster = Cluster()
    for _ in range(num_nodes):
        cluster.add_node(num_cpus=num_cpus, object_store_memory=object_store_memory)
    cluster.wait_for_nodes()
    return cluster