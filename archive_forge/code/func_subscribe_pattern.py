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
def subscribe_pattern(self, pattern: Pattern, listener=None):
    """ Subscribe to all topics matching a regex pattern.
        Subsequent calls `subscribe_from_pattern()` by Coordinator will provide
        the actual subscription topics.

        Caller: Consumer.
        Affects: SubscriptionState.subscribed_pattern
        """
    assert hasattr(pattern, 'match'), 'Expected Pattern type'
    assert listener is None or isinstance(listener, ConsumerRebalanceListener)
    self._set_subscription_type(SubscriptionType.AUTO_PATTERN)
    self._subscribed_pattern = pattern
    self._listener = listener