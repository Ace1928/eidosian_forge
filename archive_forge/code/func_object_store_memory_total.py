import logging
import os
import time
from collections import namedtuple
from numbers import Number
from typing import Any, Dict, Optional
import ray
from ray._private.resource_spec import NODE_ID_PREFIX
def object_store_memory_total(self):
    return self.object_store_memory + self.extra_object_store_memory