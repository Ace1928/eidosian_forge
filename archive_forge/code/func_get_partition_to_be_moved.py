import logging
from collections import defaultdict, namedtuple
from copy import deepcopy
def get_partition_to_be_moved(self, partition, old_consumer, new_consumer):
    if partition.topic not in self.partition_movements_by_topic:
        return partition
    if partition in self.partition_movements:
        assert old_consumer == self.partition_movements[partition].dst_member_id
        old_consumer = self.partition_movements[partition].src_member_id
    reverse_pair = ConsumerPair(src_member_id=new_consumer, dst_member_id=old_consumer)
    if reverse_pair not in self.partition_movements_by_topic[partition.topic]:
        return partition
    return next(iter(self.partition_movements_by_topic[partition.topic][reverse_pair]))