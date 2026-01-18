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
def paused_partitions(self) -> Set[TopicPartition]:
    res = set()
    for tp in self.assigned_partitions():
        if self._assigned_state(tp).paused:
            res.add(tp)
    return res