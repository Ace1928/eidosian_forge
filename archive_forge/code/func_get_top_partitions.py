from enum import Enum
from typing import NamedTuple, Dict, List, Set
from torch.fx.node import Node, map_arg
def get_top_partitions(partitions: List[Partition]) -> List[Partition]:
    """This function is to return all the partitions without parents
        as the starting points of all the paths
        """
    top_partitions = []
    for partition in partitions:
        if len(partition.parents) == 0:
            top_partitions.append(partition)
    return top_partitions