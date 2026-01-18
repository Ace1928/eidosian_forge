import collections
import copy
import logging
import threading
import time
from concurrent.futures import Future
from aiokafka import errors as Errors
from aiokafka.conn import collect_hosts
from aiokafka.structs import BrokerMetadata, PartitionMetadata, TopicPartition
def broker_metadata(self, broker_id):
    """Get BrokerMetadata

        Arguments:
            broker_id (int): node_id for a broker to check

        Returns:
            BrokerMetadata or None if not found
        """
    return self._brokers.get(broker_id) or self._bootstrap_brokers.get(broker_id) or self._coordinator_brokers.get(broker_id)