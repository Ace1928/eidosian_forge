import logging
import os
import time
from collections import namedtuple
from numbers import Number
from typing import Any, Dict, Optional
import ray
from ray._private.resource_spec import NODE_ID_PREFIX
def get_num_cpus(self) -> int:
    self.update_avail_resources()
    return self._avail_resources.cpu