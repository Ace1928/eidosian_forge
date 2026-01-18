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
def get_cache_file(mode: Mode) -> Path:
    return CACHE_DIR / f'cache.{mode.get_cache_key()}.pickle'