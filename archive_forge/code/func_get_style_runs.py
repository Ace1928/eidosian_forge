from __future__ import annotations
import re
import sys
from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING, Optional
from pyglet import event
from pyglet.text import runlist
def get_style_runs(self, attribute):
    try:
        return self._style_runs[attribute].get_run_iterator()
    except KeyError:
        return _no_style_range_iterator