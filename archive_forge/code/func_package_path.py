from __future__ import annotations # isort:skip
import hashlib
import json
from os.path import splitext
from pathlib import Path
from sys import stdout
from typing import TYPE_CHECKING, Any, TextIO
from urllib.parse import urljoin
from urllib.request import urlopen
def package_path(filename: str | Path) -> Path:
    return package_dir() / filename