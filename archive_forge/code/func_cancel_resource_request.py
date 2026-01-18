from typing import Dict, List, Optional
from dataclasses import dataclass
import ray
from ray import SCRIPT_MODE, LOCAL_MODE
from ray.air.execution.resources.request import (
from ray.air.execution.resources.resource_manager import ResourceManager
from ray.util.annotations import DeveloperAPI
def cancel_resource_request(self, resource_request: ResourceRequest):
    self._requested_resources.remove(resource_request)