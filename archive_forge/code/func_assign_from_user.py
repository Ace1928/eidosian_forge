import logging
import contextlib
import copy
import time
from asyncio import shield, Event, Future
from enum import Enum
from typing import Dict, FrozenSet, Iterable, List, Pattern, Set
from aiokafka.errors import IllegalStateError
from aiokafka.structs import OffsetAndMetadata, TopicPartition
from aiokafka.abc import ConsumerRebalanceListener
from aiokafka.util import create_future, get_running_loop
def assign_from_user(self, partitions: Iterable[TopicPartition]):
    """ Manually assign partitions. After this call automatic assignment
        will be impossible and will raise an ``IllegalStateError``.

        Caller: Consumer.
        Affects: SubscriptionState.subscription
        """
    self._set_subscription_type(SubscriptionType.USER_ASSIGNED)
    self._change_subscription(ManualSubscription(partitions, loop=self._loop))
    self._notify_assignment_waiters()