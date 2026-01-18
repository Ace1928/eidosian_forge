from __future__ import annotations
import contextlib
import sys
import weakref
from math import inf
from typing import TYPE_CHECKING, NoReturn
import pytest
from ... import _core
from .tutil import gc_collect_harder, restore_unraisablehook
def my_firstiter(agen: AsyncGenerator[object, NoReturn]) -> None:
    record.append('firstiter ' + agen.ag_frame.f_locals['arg'])