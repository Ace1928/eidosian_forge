import logging
from collections import defaultdict, namedtuple
from copy import deepcopy
from aiokafka.coordinator.assignors.abstract import AbstractPartitionAssignor
from aiokafka.coordinator.assignors.sticky.partition_movements import PartitionMovements
from aiokafka.coordinator.assignors.sticky.sorted_set import SortedSet
from aiokafka.coordinator.protocol import (
from aiokafka.coordinator.protocol import Schema
from aiokafka.protocol.struct import Struct
from aiokafka.protocol.types import String, Array, Int32
from aiokafka.structs import TopicPartition
@classmethod
def parse_member_metadata(cls, metadata):
    """
        Parses member metadata into a python object.
        This implementation only serializes and deserializes the
        StickyAssignorMemberMetadataV1 user data, since no StickyAssignor written in
        Python was deployed ever in the wild with version V0, meaning that there is no
        need to support backward compatibility with V0.

        Arguments:
          metadata (MemberMetadata): decoded metadata for a member of the group.

        Returns:
          parsed metadata (StickyAssignorMemberMetadataV1)
        """
    user_data = metadata.user_data
    if not user_data:
        return StickyAssignorMemberMetadataV1(partitions=[], generation=cls.DEFAULT_GENERATION_ID, subscription=metadata.subscription)
    try:
        decoded_user_data = StickyAssignorUserDataV1.decode(user_data)
    except Exception as e:
        log.error('Could not parse member data', e)
        return StickyAssignorMemberMetadataV1(partitions=[], generation=cls.DEFAULT_GENERATION_ID, subscription=metadata.subscription)
    member_partitions = []
    for topic, partitions in decoded_user_data.previous_assignment:
        member_partitions.extend([TopicPartition(topic, partition) for partition in partitions])
    return StickyAssignorMemberMetadataV1(partitions=member_partitions, generation=decoded_user_data.generation, subscription=metadata.subscription)