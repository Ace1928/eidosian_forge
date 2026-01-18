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
@property
def stx(self) -> Optional['StateData']:
    """
        Returns the StateData
        """
    return self.ctx.get('stx', None)