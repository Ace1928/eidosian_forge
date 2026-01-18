import asyncio
import base64
import copy
import logging
import os
import socket
import threading
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Tuple, Union
import click
import yaml
import wandb
from wandb.errors import CommError
from wandb.sdk.launch._launch_add import launch_add
from wandb.sdk.launch.errors import LaunchError
from wandb.sdk.launch.sweeps import SchedulerError
from wandb.sdk.launch.sweeps.utils import (
from wandb.sdk.launch.utils import (
from wandb.sdk.lib.runid import generate_id
@property
def at_runcap(self) -> bool:
    """False if under user-specified cap on # of runs."""
    run_cap = self._sweep_config.get('run_cap')
    if not run_cap:
        return False
    at_runcap: bool = self._num_runs_launched >= run_cap
    return at_runcap