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
def is_primary_server_process(self) -> bool:
    """
        Returns whether this is the primary server process
        """
    return self.process_id == self.stx.get('primary_server_process_id', 0)