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
def begin_reassignment(self):
    """ Signal from Coordinator that a group re-join is needed. For example
        this will be called if a commit or heartbeat fails with an
        InvalidMember error.

        Caller: Coordinator
        """
    if self._subscription is not None:
        self._subscription._begin_reassignment()