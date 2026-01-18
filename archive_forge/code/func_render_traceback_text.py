from __future__ import annotations
import itertools
import linecache
import os
import re
import sys
import sysconfig
import traceback
import typing as t
from markupsafe import escape
from ..utils import cached_property
from .console import Console
def render_traceback_text(self) -> str:
    return ''.join(self._te.format())