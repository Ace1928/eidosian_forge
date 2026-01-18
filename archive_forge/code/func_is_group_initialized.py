import logging
import os
from typing import List
import numpy as np
import ray
from ray.util.collective import types
def is_group_initialized(group_name):
    """Check if the group is initialized in this process by the group name."""
    return _group_mgr.is_group_exist(group_name)