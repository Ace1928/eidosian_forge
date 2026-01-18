import logging
from typing import Optional
import wandb
from wandb.util import (
from . import wandb_helper
from .lib import config_util
def merge_locked(self, d, user=None, _allow_val_change=None):
    """Recursively merge-update config with `d` and lock config updates on d's keys."""
    num = self._get_user_id(user)
    callback_d = {}
    for k, v in d.items():
        k, v = self._sanitize(k, v, allow_val_change=_allow_val_change)
        self._locked[k] = num
        if k in self._items and isinstance(self._items[k], dict) and isinstance(v, dict):
            self._items[k] = config_util.merge_dicts(self._items[k], v)
        else:
            self._items[k] = v
        callback_d[k] = self._items[k]
    if self._callback:
        self._callback(data=callback_d)