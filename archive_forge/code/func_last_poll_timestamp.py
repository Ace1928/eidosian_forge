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
def last_poll_timestamp(self, partition):
    """ Returns the timestamp of the last poll of this partition (in ms).
        It is the last time :meth:`highwater` and :meth:`last_stable_offset` were
        updated. However it does not mean that new messages were received.

        As with :meth:`highwater` will not be available until some messages are
        consumed.

        Arguments:
            partition (TopicPartition): partition to check

        Returns:
            int or None: timestamp if available
        """
    assert self._subscription.is_assigned(partition), 'Partition is not assigned'
    assignment = self._subscription.subscription.assignment
    return assignment.state_value(partition).timestamp