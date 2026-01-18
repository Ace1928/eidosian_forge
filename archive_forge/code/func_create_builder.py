import asyncio
import collections
import copy
import time
from aiokafka.errors import (KafkaTimeoutError,
from aiokafka.record.legacy_records import LegacyRecordBatchBuilder
from aiokafka.record.default_records import DefaultRecordBatchBuilder
from aiokafka.structs import RecordMetadata
from aiokafka.util import create_future, get_running_loop
def create_builder(self):
    if self._api_version >= (0, 11):
        magic = 2
    elif self._api_version >= (0, 10):
        magic = 1
    else:
        magic = 0
    is_transactional = False
    if self._txn_manager is not None and self._txn_manager.transactional_id is not None:
        is_transactional = True
    return BatchBuilder(magic, self._batch_size, self._compression_type, is_transactional=is_transactional)