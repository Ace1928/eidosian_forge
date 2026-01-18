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
def set_primary_server_process_id(self, process_id: Optional[int]=None):
    """
        Sets the primary server process id
        """
    if 'primary_server_process_id' in self.stx.keys():
        return
    if process_id is None and self.server_process_id_path.exists():
        with contextlib.suppress(Exception):
            process_id = int(self.server_process_id_path.read_text())
    if process_id is None:
        return
    self.stx['primary_server_process_id'] = process_id
    logger.info(f'Primary Server Process ID: {process_id}', colored=True, prefix='|g|State|e|')