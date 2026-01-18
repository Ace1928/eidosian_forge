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
def entry_at(cls, filepath: PathLike, index: int) -> 'RefLogEntry':
    """
        :return: RefLogEntry at the given index.

        :param filepath: Full path to the index file from which to read the entry.

        :param index: Python list compatible index, i.e. it may be negative to
            specify an entry counted from the end of the list.

        :raise IndexError: If the entry didn't exist.

        .. note:: This method is faster as it only parses the entry at index, skipping
            all other lines. Nonetheless, the whole file has to be read if
            the index is negative.
        """
    with open(filepath, 'rb') as fp:
        if index < 0:
            return RefLogEntry.from_line(fp.readlines()[index].strip())
        for i in range(index + 1):
            line = fp.readline()
            if not line:
                raise IndexError(f'Index file ended at line {i + 1}, before given index was reached')
        return RefLogEntry.from_line(line.strip())