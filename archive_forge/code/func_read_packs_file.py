import os
import stat
import sys
import warnings
from contextlib import suppress
from io import BytesIO
from typing import (
from .errors import NotTreeError
from .file import GitFile
from .objects import (
from .pack import (
from .protocol import DEPTH_INFINITE
from .refs import PEELED_TAG_SUFFIX, Ref
def read_packs_file(f):
    """Yield the packs listed in a packs file."""
    for line in f.read().splitlines():
        if not line:
            continue
        kind, name = line.split(b' ', 1)
        if kind != b'P':
            continue
        yield os.fsdecode(name)