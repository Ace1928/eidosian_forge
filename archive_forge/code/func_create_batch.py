import asyncio
import logging
import sys
import traceback
import warnings
from aiokafka.client import AIOKafkaClient
from aiokafka.codec import has_gzip, has_snappy, has_lz4, has_zstd
from aiokafka.errors import (
from aiokafka.partitioner import DefaultPartitioner
from aiokafka.record.default_records import DefaultRecordBatch
from aiokafka.record.legacy_records import LegacyRecordBatchBuilder
from aiokafka.structs import TopicPartition
from aiokafka.util import (
from .message_accumulator import MessageAccumulator
from .sender import Sender
from .transaction_manager import TransactionManager
def create_batch(self):
    """Create and return an empty :class:`.BatchBuilder`.

        The batch is not queued for send until submission to :meth:`send_batch`.

        Returns:
            BatchBuilder: empty batch to be filled and submitted by the caller.
        """
    return self._message_accumulator.create_builder()