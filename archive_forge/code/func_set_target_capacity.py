from typing import Any, Dict, Optional
import ray
from ray.serve._private.autoscaling_policy import BasicAutoscalingPolicy
from ray.serve._private.common import TargetCapacityDirection
from ray.serve._private.config import DeploymentConfig, ReplicaConfig
from ray.serve.generated.serve_pb2 import DeploymentInfo as DeploymentInfoProto
from ray.serve.generated.serve_pb2 import (
def set_target_capacity(self, new_target_capacity: Optional[float], new_target_capacity_direction: Optional[TargetCapacityDirection]):
    self.target_capacity = new_target_capacity
    self.target_capacity_direction = new_target_capacity_direction