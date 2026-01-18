import logging
import time
from collections import Counter
from functools import reduce
from typing import Dict, List
from ray._private.gcs_utils import PlacementGroupTableData
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.util import (
from ray.core.generated.common_pb2 import PlacementStrategy
def placement_group_deserializer(pg_tuple):
    bundles = list(map(dict, pg_tuple[0]))
    return {'bundles': freq_of_dicts(bundles), 'strategy': PlacementStrategy.Name(pg_tuple[1])}