import asyncio
import collections
import copy
import time
from aiokafka.errors import (KafkaTimeoutError,
from aiokafka.record.legacy_records import LegacyRecordBatchBuilder
from aiokafka.record.default_records import DefaultRecordBatchBuilder
from aiokafka.structs import RecordMetadata
from aiokafka.util import create_future, get_running_loop
def fail_all(self, exception):
    for batches in self._batches.values():
        for batch in batches:
            batch.failure(exception)
    for batch in self._pending_batches:
        batch.failure(exception)
    self._exception = exception