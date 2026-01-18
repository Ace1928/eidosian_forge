import asyncio
import collections
import logging
import random
import time
from itertools import chain
import async_timeout
import aiokafka.errors as Errors
from aiokafka.errors import (
from aiokafka.protocol.offset import OffsetRequest
from aiokafka.protocol.fetch import FetchRequest
from aiokafka.record.memory_records import MemoryRecords
from aiokafka.record.control_record import ControlRecord, ABORT_MARKER
from aiokafka.structs import OffsetAndTimestamp, TopicPartition, ConsumerRecord
from aiokafka.util import create_future, create_task
def request_offset_reset(self, tps, strategy):
    """ Force a position reset. Called from Consumer of `seek_to_*` API's.
        """
    assignment = self._subscriptions.subscription.assignment
    assert assignment is not None
    waiters = []
    for tp in tps:
        tp_state = assignment.state_value(tp)
        tp_state.await_reset(strategy)
        waiters.append(tp_state.wait_for_position())
        if tp in self._records:
            del self._records[tp]
    self._notify(self._wait_consume_future)
    return asyncio.gather(*waiters)