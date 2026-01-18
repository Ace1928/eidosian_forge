import asyncio
import collections
import logging
import copy
import time
import aiokafka.errors as Errors
from aiokafka.client import ConnectionGroup, CoordinationType
from aiokafka.coordinator.assignors.roundrobin import RoundRobinPartitionAssignor
from aiokafka.coordinator.protocol import ConsumerProtocol
from aiokafka.protocol.api import Response
from aiokafka.protocol.commit import (
from aiokafka.protocol.group import (
from aiokafka.structs import OffsetAndMetadata, TopicPartition
from aiokafka.util import create_future, create_task
def coordinator_dead(self):
    """ Mark the current coordinator as dead.
        NOTE: this will not force a group rejoin. If new coordinator is able to
        recognize this member we will just continue with current generation.
        """
    if self.coordinator_id is not None:
        log.warning('Marking the coordinator dead (node %s)for group %s.', self.coordinator_id, self.group_id)
        self.coordinator_id = None
        self._coordinator_dead_fut.set_result(None)