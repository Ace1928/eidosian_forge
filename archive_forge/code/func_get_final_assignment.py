import logging
from collections import defaultdict, namedtuple
from copy import deepcopy
from aiokafka.coordinator.assignors.abstract import AbstractPartitionAssignor
from aiokafka.coordinator.assignors.sticky.partition_movements import PartitionMovements
from aiokafka.coordinator.assignors.sticky.sorted_set import SortedSet
from aiokafka.coordinator.protocol import (
from aiokafka.coordinator.protocol import Schema
from aiokafka.protocol.struct import Struct
from aiokafka.protocol.types import String, Array, Int32
from aiokafka.structs import TopicPartition
def get_final_assignment(self, member_id):
    assignment = defaultdict(list)
    for topic_partition in self.current_assignment[member_id]:
        assignment[topic_partition.topic].append(topic_partition.partition)
    assignment = {k: sorted(v) for k, v in assignment.items()}
    return assignment.items()