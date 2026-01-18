import logging
import os
import time
from collections import namedtuple
from numbers import Number
from typing import Any, Dict, Optional
import ray
from ray._private.resource_spec import NODE_ID_PREFIX
def is_nonnegative(self):
    all_values = [self.cpu, self.gpu, self.extra_cpu, self.extra_gpu]
    all_values += list(self.custom_resources.values())
    all_values += list(self.extra_custom_resources.values())
    return all((v >= 0 for v in all_values))