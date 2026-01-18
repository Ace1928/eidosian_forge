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
def detect_colors(self):
    """Update color output settings.

        Parse the `color_mode` string and optionally disable or force-enable
        color output
        (8-color ANSI if no terminal detected to be safe) in colorful.
        """
    if self.color_mode == 'true':
        if self._autodetected_cf_colormode != cf.NO_COLORS:
            cf.colormode = self._autodetected_cf_colormode
        else:
            cf.colormode = cf.ANSI_8_COLORS
        return
    if self.color_mode == 'false':
        cf.disable()
        return
    if self.color_mode == 'auto':
        return
    raise ValueError('Invalid log color setting: ' + self.color_mode)