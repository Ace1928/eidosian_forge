import inspect
import logging
import os
import sys
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple
import click
import colorama
import ray  # noqa: F401
def panic(self, *args, **kwargs):
    self._error(*args, _level_str='PANIC', **kwargs)