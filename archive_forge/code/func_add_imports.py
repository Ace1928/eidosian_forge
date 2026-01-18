from __future__ import annotations
import ast
import os
import re
import typing as t
from ..io import (
from ..util import (
from ..data import (
from ..target import (
def add_imports(self, names: list[str], line_no: int) -> None:
    """Add the given import names if they are module_utils imports."""
    for name in names:
        if self.is_module_util_name(name):
            self.add_import(name, line_no)