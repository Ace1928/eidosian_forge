from __future__ import annotations
import os
from collections import deque
from queue import Empty
from queue import LifoQueue as _LifoQueue
from typing import TYPE_CHECKING
from . import exceptions
from .utils.compat import register_after_fork
from .utils.functional import lazy
def collect_resource(self, resource):
    pass