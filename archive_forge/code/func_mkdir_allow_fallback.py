import contextlib
import ctypes
import errno
import logging
import os
import platform
import re
import shutil
import tempfile
import threading
from pathlib import Path
from typing import IO, Any, BinaryIO, Generator, Optional
from wandb.sdk.lib.paths import StrPath
def mkdir_allow_fallback(dir_name: StrPath) -> StrPath:
    """Create `dir_name`, removing invalid path characters if necessary.

    Returns:
        The path to the created directory, which may not be the original path.
    """
    for new_name in path_fallbacks(dir_name):
        try:
            os.makedirs(new_name, exist_ok=True)
            if Path(new_name) != Path(dir_name):
                logger.warning(f"Creating '{new_name}' instead of '{dir_name}'")
            return Path(new_name) if isinstance(dir_name, Path) else new_name
        except (ValueError, NotADirectoryError):
            pass
        except OSError as e:
            if e.errno != 22:
                raise
    raise OSError(f"Unable to create directory '{dir_name}'")