import collections
import copy
import logging
import threading
import time
from concurrent.futures import Future
from aiokafka import errors as Errors
from aiokafka.conn import collect_hosts
from aiokafka.structs import BrokerMetadata, PartitionMetadata, TopicPartition
def partitions_for_broker(self, broker_id):
    """Return TopicPartitions for which the broker is a leader.

        Arguments:
            broker_id (int): node id for a broker

        Returns:
            set: {TopicPartition, ...}
            None if the broker either has no partitions or does not exist.
        """
    return self._broker_partitions.get(broker_id)