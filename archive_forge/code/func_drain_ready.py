import asyncio
import collections
import copy
import time
from aiokafka.errors import (KafkaTimeoutError,
from aiokafka.record.legacy_records import LegacyRecordBatchBuilder
from aiokafka.record.default_records import DefaultRecordBatchBuilder
from aiokafka.structs import RecordMetadata
from aiokafka.util import create_future, get_running_loop
def drain_ready(self):
    """Compress batch to be ready for send"""
    if not self._drain_waiter.done():
        self._drain_waiter.set_result(None)
    self._retry_count += 1