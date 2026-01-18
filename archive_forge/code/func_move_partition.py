import logging
from collections import defaultdict, namedtuple
from copy import deepcopy
def move_partition(self, partition, old_consumer, new_consumer):
    pair = ConsumerPair(src_member_id=old_consumer, dst_member_id=new_consumer)
    if partition in self.partition_movements:
        existing_pair = self._remove_movement_record_of_partition(partition)
        assert existing_pair.dst_member_id == old_consumer
        if existing_pair.src_member_id != new_consumer:
            self._add_partition_movement_record(partition, ConsumerPair(src_member_id=existing_pair.src_member_id, dst_member_id=new_consumer))
    else:
        self._add_partition_movement_record(partition, pair)