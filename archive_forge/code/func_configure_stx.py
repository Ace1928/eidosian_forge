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
def configure_stx(self):
    """
        Configures the stateful statefuldata
        """
    if 'stx' not in self.ctx:
        stx_filepath = self.data_path.joinpath(f'{self.app_module_name}.state.json')
        self.ctx['stx_filepath'] = stx_filepath.as_posix()
        stx = StateData(filepath=stx_filepath)
        self.ctx['stx'] = stx
        atexit.register(self.on_exit)