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
def add_leader_process_id(self, process_id: int, kind: Optional[str]=None):
    """
        Adds a leader process id
        """
    self.stx.append('leader_process_ids', process_id)