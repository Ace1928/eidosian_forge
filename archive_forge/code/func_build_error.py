import logging
import os
from typing import Dict, List, Optional
import ray._private.ray_constants as ray_constants
from ray._private.utils import (
def build_error(resource, alternative):
    return f"{self.resources} -> `{resource}` cannot be a custom resource because it is one of the default resources ({ray_constants.DEFAULT_RESOURCES}). Use `{alternative}` instead. For example, use `ray start --{alternative.replace('_', '-')}=1` instead of `ray start --resources={{'{resource}': 1}}`"