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
def print_actions(self) -> None:
    for action, line in self._log_actions:
        self._info(f'{action.capitalize()} ({line})')
    self._log_actions = []