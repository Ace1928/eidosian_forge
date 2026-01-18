import errno
import itertools
import logging
import os.path
import tempfile
import traceback
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import (
from pip._internal.utils.misc import enum, rmtree
@contextmanager
def global_tempdir_manager() -> Generator[None, None, None]:
    global _tempdir_manager
    with ExitStack() as stack:
        old_tempdir_manager, _tempdir_manager = (_tempdir_manager, stack)
        try:
            yield
        finally:
            _tempdir_manager = old_tempdir_manager