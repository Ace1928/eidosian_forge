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
def requesting_committed(self):
    """ Return all partitions that are requesting commit point fetch """
    requesting = []
    for tp in self._topic_partitions:
        tp_state = self.state_value(tp)
        if tp_state._committed_futs:
            requesting.append(tp)
    return requesting