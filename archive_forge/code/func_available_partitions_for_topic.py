import collections
import copy
import logging
import threading
import time
from concurrent.futures import Future
from aiokafka import errors as Errors
from aiokafka.conn import collect_hosts
from aiokafka.structs import BrokerMetadata, PartitionMetadata, TopicPartition
def available_partitions_for_topic(self, topic):
    """Return set of partitions with known leaders

        Arguments:
            topic (str): topic to check for partitions

        Returns:
            set: {partition (int), ...}
            None if topic not found.
        """
    if topic not in self._partitions:
        return None
    return set([partition for partition, metadata in self._partitions[topic].items() if metadata.leader != -1])