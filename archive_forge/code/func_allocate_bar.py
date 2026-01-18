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
def allocate_bar(self, state: ProgressBarState) -> None:
    """Add a new bar to this group."""
    self.bars_by_uuid[state['uuid']] = _Bar(state, self.pos_offset)