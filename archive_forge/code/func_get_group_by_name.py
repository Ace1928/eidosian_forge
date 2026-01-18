import logging
import os
from typing import List
import numpy as np
import ray
from ray.util.collective import types
def get_group_by_name(self, group_name):
    """Get the collective group handle by its name."""
    if not self.is_group_exist(group_name):
        logger.warning("The group '{}' is not initialized.".format(group_name))
        return None
    return self._name_group_map[group_name]