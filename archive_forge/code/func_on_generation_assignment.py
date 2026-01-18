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
@classmethod
def on_generation_assignment(cls, generation):
    """Callback that runs on each assignment. Updates assignor's generation id.

        Arguments:
          generation: generation id
        """
    log.debug('On generation assignment: generation={}'.format(generation))
    cls.generation = generation