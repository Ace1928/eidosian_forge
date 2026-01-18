from __future__ import annotations
import os
import stat
import sys
from errno import EACCES, EISDIR
from pathlib import Path
def ensure_directory_exists(filename: Path | str) -> None:
    """
    Ensure the directory containing the file exists (create it if necessary)
    :param filename: file.
    """
    Path(filename).parent.mkdir(parents=True, exist_ok=True)