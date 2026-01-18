import time
from typing import Any, Callable, Iterable, List, Tuple, Union
import ray
from ray import ObjectRef
from ray.cluster_utils import Cluster
def input_reader(i: PartitionID) -> Iterable[InType]:
    for _ in range(num_partitions):
        yield np.ones((rows_per_partition // num_partitions, 2), dtype=np.int64)