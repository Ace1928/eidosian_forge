import builtins
import copy
import json
import logging
import os
import sys
import threading
import uuid
from typing import Any, Dict, Iterable, Optional
import colorama
import ray
from ray._private.ray_constants import env_bool
from ray.util.debug import log_once
def has_bar(self, bar_uuid) -> bool:
    """Return whether this bar exists."""
    return bar_uuid in self.bars_by_uuid