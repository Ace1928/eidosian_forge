import copy
import inspect
import io
import re
import warnings
from configparser import (
from dataclasses import dataclass
from pathlib import Path
from types import GeneratorType
from typing import (
import srsly
from .util import SimpleFrozenDict, SimpleFrozenList  # noqa: F401
def alias_generator(name: str) -> str:
    """Generate field aliases in promise schema."""
    if name == ARGS_FIELD_ALIAS:
        return ARGS_FIELD
    if name in RESERVED_FIELDS:
        return RESERVED_FIELDS[name]
    return name