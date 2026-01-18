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
def safe_copy(source_path: StrPath, target_path: StrPath) -> StrPath:
    """Copy a file, ensuring any changes only apply atomically once finished."""
    output_path = Path(target_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_path.parent) as tmp_dir:
        tmp_path = (Path(tmp_dir) / Path(source_path).name).with_suffix('.tmp')
        shutil.copy2(source_path, tmp_path)
        tmp_path.replace(output_path)
    return target_path