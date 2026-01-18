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
def all_consumed_offsets(self) -> Dict[TopicPartition, OffsetAndMetadata]:
    """ Returns consumed offsets as {TopicPartition: OffsetAndMetadata} """
    all_consumed = {}
    for tp in self._topic_partitions:
        state = self.state_value(tp)
        if state.has_valid_position:
            all_consumed[tp] = OffsetAndMetadata(state.position, '')
    return all_consumed