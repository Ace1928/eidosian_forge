import abc
import logging
import threading
import time
import traceback
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from ray._private.thirdparty.tabulate.tabulate import tabulate
from ray.train.constants import _DEPRECATED_VALUE
from ray.util import log_once
from ray.util.annotations import PublicAPI
from ray.widgets import Template
def retry(self):
    if not self._current_cmd:
        raise RuntimeError('No sync command set, cannot retry.')
    cmd, kwargs = self._current_cmd
    self._sync_process = _BackgroundProcess(cmd)
    self._sync_process.start(**kwargs)