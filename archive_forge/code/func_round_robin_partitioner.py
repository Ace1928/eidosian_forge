import time
from typing import Any, Callable, Iterable, List, Tuple, Union
import ray
from ray import ObjectRef
from ray.cluster_utils import Cluster
def round_robin_partitioner(input_stream: Iterable[InType], num_partitions: int) -> Iterable[Tuple[PartitionID, InType]]:
    """Round robin partitions items from the input reader.

    You can write custom partitioning functions for your use case.

    Args:
        input_stream: Iterator over items from the input reader.
        num_partitions: Number of output partitions.

    Yields:
        Tuples of (partition id, input item).
    """
    i = 0
    for item in input_stream:
        yield (i, item)
        i += 1
        i %= num_partitions