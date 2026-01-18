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
def process_state_update(self, state: ProgressBarState) -> None:
    """Apply the remote progress bar state update.

        This creates a new bar locally if it doesn't already exist. When a bar is
        created or destroyed, we also recalculate and update the `pos_offset` of each
        BarGroup on the screen.
        """
    with self.lock:
        self._process_state_update_locked(state)