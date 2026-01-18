import logging
import math
from abc import ABCMeta, abstractmethod
from typing import List, Optional
from ray.serve._private.common import TargetCapacityDirection
from ray.serve._private.constants import CONTROL_LOOP_PERIOD_S, SERVE_LOGGER_NAME
from ray.serve._private.utils import get_capacity_adjusted_num_replicas
from ray.serve.config import AutoscalingConfig
def get_current_lower_bound(self, target_capacity: Optional[float]=None, target_capacity_direction: Optional[TargetCapacityDirection]=None) -> int:
    """Get the autoscaling lower bound, including target_capacity changes.

        The autoscaler uses initial_replicas scaled by target_capacity only
        if the target capacity direction is UP.
        """
    if self.config.initial_replicas is not None and target_capacity_direction == TargetCapacityDirection.UP:
        return get_capacity_adjusted_num_replicas(self.config.initial_replicas, target_capacity)
    else:
        return get_capacity_adjusted_num_replicas(self.config.min_replicas, target_capacity)