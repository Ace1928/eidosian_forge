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
@property
def fetcher_idle_time(self):
    """ How much time (in seconds) spent without consuming any records """
    if self._fetch_count == 0:
        return time.monotonic() - self._last_fetch_ended
    else:
        return 0