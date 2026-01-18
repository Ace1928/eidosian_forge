from __future__ import annotations
from dataclasses import dataclass, field
import re
import codecs
import os
import typing as T
from .mesonlib import MesonException
from . import mlog
def set_kwarg(self, name: IdNode, value: BaseNode) -> None:
    if any((isinstance(x, IdNode) and name.value == x.value for x in self.kwargs)):
        mlog.warning(f'Keyword argument "{name.value}" defined multiple times.', location=self)
        mlog.warning('This will be an error in future Meson releases.')
    self.kwargs[name] = value