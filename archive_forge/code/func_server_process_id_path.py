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
def server_process_id_path(self) -> pathlib.Path:
    """
        Returns the server process id path
        """
    return self.data_path.joinpath(f'{self.app_module_name}.pid')