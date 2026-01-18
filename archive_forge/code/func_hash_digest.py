import hashlib
import os
import pickle
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, NamedTuple, Set, Tuple
from platformdirs import user_cache_dir
from _black_version import version as __version__
from black.mode import Mode
from black.output import err
@staticmethod
def hash_digest(path: Path) -> str:
    """Return hash digest for path."""
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest()