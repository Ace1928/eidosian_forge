from collections import defaultdict
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict
from ray.tune.callback import Callback
from ray.tune.experiment import Trial
def all_trials_are_terminated(self) -> bool:
    """True if all trials are terminated."""
    if not self._snapshot:
        return False
    last_snapshot = self._snapshot[-1]
    return all((last_snapshot[trial_id] == Trial.TERMINATED for trial_id in last_snapshot))