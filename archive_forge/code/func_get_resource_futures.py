import time
from collections import defaultdict
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
import ray
from ray.air.execution.resources.request import (
from ray.air.execution.resources.resource_manager import ResourceManager
from ray.util.annotations import DeveloperAPI
from ray.util.placement_group import PlacementGroup, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
def get_resource_futures(self) -> List[ray.ObjectRef]:
    return list(self._staging_future_to_pg.keys())