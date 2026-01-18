from __future__ import annotations
import ast
import builtins
import sys
import typing
from ast import (
from collections import defaultdict
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, ClassVar, cast, overload
def get_import(self, module: str, name: str) -> Name:
    if module in self.load_names and name in self.load_names[module]:
        return self.load_names[module][name]
    qualified_name = f'{module}.{name}'
    if name in self.imported_names and self.imported_names[name] == qualified_name:
        return Name(id=name, ctx=Load())
    alias = self.get_unused_name(name)
    node = self.load_names[module][name] = Name(id=alias, ctx=Load())
    self.imported_names[name] = qualified_name
    return node