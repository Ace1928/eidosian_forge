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
def balance(self):
    self._initialize_current_subscriptions()
    initializing = len(self.current_assignment[self._get_consumer_with_most_subscriptions()]) == 0
    for partition in self.unassigned_partitions:
        if not self.partition_to_all_potential_consumers[partition]:
            continue
        self._assign_partition(partition)
    fixed_partitions = set()
    for partition in self.partition_to_all_potential_consumers.keys():
        if not self._can_partition_participate_in_reassignment(partition):
            fixed_partitions.add(partition)
    for fixed_partition in fixed_partitions:
        remove_if_present(self.sorted_partitions, fixed_partition)
        remove_if_present(self.unassigned_partitions, fixed_partition)
    fixed_assignments = {}
    for consumer in self.consumer_to_all_potential_partitions.keys():
        if not self._can_consumer_participate_in_reassignment(consumer):
            self._remove_consumer_from_current_subscriptions_and_maintain_order(consumer)
            fixed_assignments[consumer] = self.current_assignment[consumer]
            del self.current_assignment[consumer]
    prebalance_assignment = deepcopy(self.current_assignment)
    prebalance_partition_consumers = deepcopy(self.current_partition_consumer)
    if not self.revocation_required:
        self._perform_reassignments(self.unassigned_partitions)
    reassignment_performed = self._perform_reassignments(self.sorted_partitions)
    if not initializing and reassignment_performed and (self._get_balance_score(self.current_assignment) >= self._get_balance_score(prebalance_assignment)):
        self.current_assignment = prebalance_assignment
        self.current_partition_consumer.clear()
        self.current_partition_consumer.update(prebalance_partition_consumers)
    for consumer, partitions in fixed_assignments.items():
        self.current_assignment[consumer] = partitions
        self._add_consumer_to_current_subscriptions_and_maintain_order(consumer)