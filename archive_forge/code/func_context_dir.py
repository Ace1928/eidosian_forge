from __future__ import absolute_import
import pathlib
import contextlib
from typing import Optional, Union, Dict, TYPE_CHECKING
from lazyops.types.models import BaseSettings
from lazyops.types.classprops import lazyproperty
from lazyops.imports._fileio import (
from lazyops.imports._aiokeydb import (
@lazyproperty
def context_dir(self) -> str:
    if self.in_colab:
        return '/content'
    return '/' if self.is_remote else pathlib.Path.cwd().as_posix()