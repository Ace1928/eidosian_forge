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
def assigned_partitions(self) -> Set[TopicPartition]:
    if self._subscription is None:
        return set()
    if self._subscription.assignment is None:
        return set()
    return self._subscription.assignment.tps