from typing import (
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import IterDataPipe
from enum import IntEnum
def apply_sharding(self, num_of_instances, instance_id, sharding_group=SHARDING_PRIORITIES.DEFAULT):
    if instance_id >= num_of_instances:
        raise ValueError(f'instance_id({instance_id}) should be smaller than num_of_instances({num_of_instances})')
    if sharding_group == SHARDING_PRIORITIES.DEFAULT:
        if len(self.groups) and SHARDING_PRIORITIES.DEFAULT not in self.groups:
            raise Exception('ShardingFilter cannot mix DEFAULT and non DEFAULT groups')
    elif SHARDING_PRIORITIES.DEFAULT in self.groups:
        raise Exception('ShardingFilter cannot mix DEFAULT and non DEFAULT groups')
    self.groups[sharding_group] = (num_of_instances, instance_id)
    self._update_num_of_instances()