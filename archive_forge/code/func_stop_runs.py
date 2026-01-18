import json
import os
import random
import string
import time
from typing import Callable, Dict, List, Optional, Tuple, Union
import yaml
from wandb import env
from wandb.apis import InternalApi
from wandb.sdk import wandb_sweep
from wandb.sdk.launch.sweeps.utils import (
from wandb.util import get_module
def stop_runs(self, runs: List[sweeps.SweepRun]) -> None:
    earlystop_list = list({run.name for run in runs})
    self._log_actions.append(('stop', ','.join(earlystop_list)))
    self._controller['earlystop'] = earlystop_list
    self._sweep_object_sync_to_backend()