import time
from typing import Any, Callable, Iterable, List, Tuple, Union
import ray
from ray import ObjectRef
from ray.cluster_utils import Cluster
@ray.remote(num_returns=output_num_partitions)
def shuffle_map(i: PartitionID) -> List[List[Union[Any, ObjectRef]]]:
    writers = [object_store_writer() for _ in range(output_num_partitions)]
    for out_i, item in partitioner(input_reader(i), output_num_partitions):
        writers[out_i].add(item)
    return [c.finish() for c in writers]