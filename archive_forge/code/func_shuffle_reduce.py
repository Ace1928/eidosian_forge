import time
from typing import Any, Callable, Iterable, List, Tuple, Union
import ray
from ray import ObjectRef
from ray.cluster_utils import Cluster
@ray.remote
def shuffle_reduce(i: PartitionID, *mapper_outputs: List[List[Union[Any, ObjectRef]]]) -> OutType:
    input_objects = []
    assert len(mapper_outputs) == input_num_partitions
    for obj_refs in mapper_outputs:
        for obj_ref in obj_refs:
            input_objects.append(obj_ref)
    return output_writer(i, input_objects)