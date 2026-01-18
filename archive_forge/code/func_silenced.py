from __future__ import annotations
import logging # isort:skip
import contextlib
from typing import (
from ...model import Model
from ...settings import settings
from ...util.dataclasses import dataclass, field
from .issue import Warning
@contextlib.contextmanager
def silenced(warning: Warning) -> Iterator[None]:
    silence(warning, True)
    try:
        yield
    finally:
        silence(warning, False)