from __future__ import annotations
from dataclasses import dataclass, field
import re
import codecs
import os
import typing as T
from .mesonlib import MesonException
from . import mlog
def set_kwarg_no_check(self, name: BaseNode, value: BaseNode) -> None:
    self.kwargs[name] = value