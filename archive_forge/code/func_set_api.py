import asyncio
import functools
import inspect
import logging
import sys
from typing import Any, Dict, Optional, Sequence, TypeVar
import wandb.sdk
import wandb.util
from wandb.sdk.lib import telemetry as wb_telemetry
from wandb.sdk.lib.timer import Timer
@property
def set_api(self) -> Any:
    """Returns the API module."""
    lib_name = self.name.lower()
    if self._api is None:
        self._api = wandb.util.get_module(name=lib_name, required=f'To use the W&B {self.name} Autolog, you need to have the `{lib_name}` python package installed. Please install it with `pip install {lib_name}`.', lazy=False)
    return self._api