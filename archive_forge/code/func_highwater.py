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
def highwater(self, partition):
    """ Last known highwater offset for a partition.

        A highwater offset is the offset that will be assigned to the next
        message that is produced. It may be useful for calculating lag, by
        comparing with the reported position. Note that both position and
        highwater refer to the *next* offset – i.e., highwater offset is one
        greater than the newest available message.

        Highwater offsets are returned as part of ``FetchResponse``, so will
        not be available if messages for this partition were not requested yet.

        Arguments:
            partition (TopicPartition): partition to check

        Returns:
            int or None: offset if available
        """
    assert self._subscription.is_assigned(partition), 'Partition is not assigned'
    assignment = self._subscription.subscription.assignment
    return assignment.state_value(partition).highwater