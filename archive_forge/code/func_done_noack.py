import asyncio
import collections
import copy
import time
from aiokafka.errors import (KafkaTimeoutError,
from aiokafka.record.legacy_records import LegacyRecordBatchBuilder
from aiokafka.record.default_records import DefaultRecordBatchBuilder
from aiokafka.structs import RecordMetadata
from aiokafka.util import create_future, get_running_loop
def done_noack(self):
    """ Resolve all pending futures to None """
    if not self.future.done():
        self.future.set_result(None)
    for future, _ in self._msg_futures:
        if future.done():
            continue
        future.set_result(None)