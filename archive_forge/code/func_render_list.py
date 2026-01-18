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
def render_list(self, xs: List[str], separator: str=cf.reset(', ')):
    """Render a list of bolded values using a non-bolded separator."""
    return separator.join([str(cf.bold(x)) for x in xs])