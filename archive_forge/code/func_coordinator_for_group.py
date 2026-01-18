import collections
import copy
import logging
import threading
import time
from concurrent.futures import Future
from aiokafka import errors as Errors
from aiokafka.conn import collect_hosts
from aiokafka.structs import BrokerMetadata, PartitionMetadata, TopicPartition
def coordinator_for_group(self, group):
    """Return node_id of group coordinator.

        Arguments:
            group (str): name of consumer group

        Returns:
            int: node_id for group coordinator
            None if the group does not exist.
        """
    return self._groups.get(group)