from __future__ import annotations
import os
import abc
import atexit
import pathlib
import filelock
import contextlib
from lazyops.types import BaseModel, Field
from lazyops.utils.logs import logger
from lazyops.utils.serialization import Json
from typing import Optional, Dict, Any, Set, List, Union, Generator, TYPE_CHECKING
def has_logged(self, key: str) -> bool:
    """
        Returns whether the key has been logged
        """
    return self.stx.append('logged', key)