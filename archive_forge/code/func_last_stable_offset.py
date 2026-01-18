import asyncio
import logging
import re
import sys
import traceback
import warnings
from typing import Dict, List
from aiokafka.abc import ConsumerRebalanceListener
from aiokafka.client import AIOKafkaClient
from aiokafka.coordinator.assignors.roundrobin import RoundRobinPartitionAssignor
from aiokafka.errors import (
from aiokafka.structs import TopicPartition, ConsumerRecord
from aiokafka.util import (
from aiokafka import __version__
from .fetcher import Fetcher, OffsetResetStrategy
from .group_coordinator import GroupCoordinator, NoGroupCoordinator
from .subscription_state import SubscriptionState
def last_stable_offset(self, partition):
    """ Returns the Last Stable Offset of a topic. It will be the last
        offset up to which point all transactions were completed. Only
        available in with isolation_level `read_committed`, in
        `read_uncommitted` will always return -1. Will return None for older
        Brokers.

        As with :meth:`highwater` will not be available until some messages are
        consumed.

        Arguments:
            partition (TopicPartition): partition to check

        Returns:
            int or None: offset if available
        """
    assert self._subscription.is_assigned(partition), 'Partition is not assigned'
    assignment = self._subscription.subscription.assignment
    return assignment.state_value(partition).lso