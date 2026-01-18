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
def configure_stopping(self, stopping: Union[str, Callable[[Union[dict, sweeps.SweepConfig], List[sweeps.SweepRun]], List[sweeps.SweepRun]]], **kwargs):
    self._configure_check()
    if isinstance(stopping, str):
        self._create.setdefault('early_terminate', {})
        self._create['early_terminate']['type'] = stopping
        for k, v in kwargs.items():
            self._create['early_terminate'][k] = v
    elif callable(stopping):
        self._custom_stopping = stopping(kwargs)
        self._create.setdefault('early_terminate', {})
        self._create['early_terminate']['type'] = 'custom'
    else:
        raise ControllerError('Unhandled stopping type.')