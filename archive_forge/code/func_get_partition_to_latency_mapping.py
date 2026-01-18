from enum import Enum
from typing import NamedTuple, Dict, List, Set
from torch.fx.node import Node, map_arg
def get_partition_to_latency_mapping(partitions: List[Partition], node_to_latency_mapping: Dict[Node, NodeLatency]) -> Dict[Partition, PartitionLatency]:
    """Given all the partitions and node_to_latency_mapping dictionary,
    return a mapping dictionary of each partition to its overall latency
    """
    partition_to_latency_mapping: Dict[Partition, PartitionLatency] = {}
    for partition in partitions:
        partition_latency = get_latency_of_one_partition(partition, node_to_latency_mapping)
        partition_to_latency_mapping[partition] = partition_latency
    return partition_to_latency_mapping