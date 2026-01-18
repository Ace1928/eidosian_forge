import logging
import os
import time
from collections import namedtuple
from numbers import Number
from typing import Any, Dict, Optional
import ray
from ray._private.resource_spec import NODE_ID_PREFIX
def cpu_total(self):
    return self.cpu + self.extra_cpu