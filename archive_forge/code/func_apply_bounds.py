import logging
import math
from abc import ABCMeta, abstractmethod
from typing import List, Optional
from ray.serve._private.common import TargetCapacityDirection
from ray.serve._private.constants import CONTROL_LOOP_PERIOD_S, SERVE_LOGGER_NAME
from ray.serve._private.utils import get_capacity_adjusted_num_replicas
from ray.serve.config import AutoscalingConfig
def apply_bounds(self, curr_target_num_replicas: int, target_capacity: Optional[float]=None, target_capacity_direction: Optional[TargetCapacityDirection]=None) -> int:
    """Clips curr_target_num_replicas using the current bounds."""
    upper_bound = get_capacity_adjusted_num_replicas(self.config.max_replicas, target_capacity)
    lower_bound = self.get_current_lower_bound(target_capacity, target_capacity_direction)
    return max(lower_bound, min(upper_bound, curr_target_num_replicas))