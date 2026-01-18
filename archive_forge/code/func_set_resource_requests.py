import logging
import time
from collections import Counter
from functools import reduce
from typing import Dict, List
from ray._private.gcs_utils import PlacementGroupTableData
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.util import (
from ray.core.generated.common_pb2 import PlacementStrategy
def set_resource_requests(self, requested_resources):
    if requested_resources is not None:
        assert isinstance(requested_resources, list), requested_resources
    self.resource_requests = [request for request in requested_resources if len(request) > 0]