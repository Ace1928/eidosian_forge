import logging
import time
from collections import Counter
from functools import reduce
from typing import Dict, List
from ray._private.gcs_utils import PlacementGroupTableData
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.util import (
from ray.core.generated.common_pb2 import PlacementStrategy
def format_resource(key, value):
    if key in ['object_store_memory', 'memory']:
        return '{} GiB'.format(round(value / (1024 * 1024 * 1024), 2))
    else:
        return round(value, 2)