import base64
import json
import os
from copy import deepcopy
from ..optimizer import AcceleratedOptimizer
from ..scheduler import AcceleratedScheduler
@property
def step_was_skipped(self):
    """Whether or not the optimizer step was done, or skipped because of gradient overflow."""
    if self.__has_overflow__:
        return self.optimizer.overflow
    return False