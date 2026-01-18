from __future__ import annotations
import os
import abc
import atexit
import tempfile
import pathlib
import filelock
import contextlib
from lazyops.utils.serialization import Json
from typing import Optional, Dict, Any, Set, List, Union, Generator, TYPE_CHECKING
def cleanup_on_exit(self):
    """
        Cleans up on exit
        """
    if not self.filepath.exists() and (not self.filelock_path.exists()):
        return
    if self.is_multithreaded and self['process_id'] != os.getpid():
        return
    with contextlib.suppress(Exception):
        self.close()
        self.filepath.unlink()
        self.filelock_path.unlink()