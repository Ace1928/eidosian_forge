import base64
import json
import os
from copy import deepcopy
from ..optimizer import AcceleratedOptimizer
from ..scheduler import AcceleratedScheduler
def is_true(self, ds_key_long):
    """
        Returns `True`/``False` only if the value is set, always `False` otherwise. So use this method to ask the very
        specific question of whether the value is set to `True` (and it's not set to `False`` or isn't set).

        """
    value = self.get_value(ds_key_long)
    return False if value is None else bool(value)