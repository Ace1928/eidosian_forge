from typing import Dict, List, Optional
from dataclasses import dataclass
import ray
from ray import SCRIPT_MODE, LOCAL_MODE
from ray.air.execution.resources.request import (
from ray.air.execution.resources.resource_manager import ResourceManager
from ray.util.annotations import DeveloperAPI
def free_resources(self, acquired_resource: AcquiredResources):
    resources = acquired_resource.resource_request
    self._used_resources.remove(resources)