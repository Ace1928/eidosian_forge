import base64
import json
import os
from copy import deepcopy
from ..optimizer import AcceleratedOptimizer
from ..scheduler import AcceleratedScheduler
def is_zero2(self):
    return self._stage == 2