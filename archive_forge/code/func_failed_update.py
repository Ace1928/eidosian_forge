import collections
import copy
import logging
import threading
import time
from concurrent.futures import Future
from aiokafka import errors as Errors
from aiokafka.conn import collect_hosts
from aiokafka.structs import BrokerMetadata, PartitionMetadata, TopicPartition
def failed_update(self, exception):
    """Update cluster state given a failed MetadataRequest."""
    f = None
    with self._lock:
        if self._future:
            f = self._future
            self._future = None
    if f:
        f.failure(exception)
    self._last_refresh_ms = time.time() * 1000