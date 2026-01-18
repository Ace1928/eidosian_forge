from mmap import mmap
import re
import time as _time
from git.compat import defenc
from git.objects.util import (
from git.util import (
import os.path as osp
from typing import Iterator, List, Tuple, Union, TYPE_CHECKING
from git.types import PathLike
@property
def oldhexsha(self) -> str:
    """The hexsha to the commit the ref pointed to before the change."""
    return self[0]