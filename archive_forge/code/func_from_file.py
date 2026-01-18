from mmap import mmap
import re
import time as _time
from git.compat import defenc
from git.objects.util import (
from git.util import (
import os.path as osp
from typing import Iterator, List, Tuple, Union, TYPE_CHECKING
from git.types import PathLike
@classmethod
def from_file(cls, filepath: PathLike) -> 'RefLog':
    """
        :return: A new RefLog instance containing all entries from the reflog
            at the given filepath
        :param filepath: Path to reflog
        :raise ValueError: If the file could not be read or was corrupted in some way
        """
    return cls(filepath)