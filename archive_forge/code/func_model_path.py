from __future__ import absolute_import
import pathlib
import contextlib
from typing import Optional, Union, Dict, TYPE_CHECKING
from lazyops.types.models import BaseSettings
from lazyops.types.classprops import lazyproperty
from lazyops.imports._fileio import (
from lazyops.imports._aiokeydb import (
@lazyproperty
def model_path(self) -> FileLike:
    model_dir = self.model_dir or self.data_path.joinpath('models')
    return File(model_dir) if _fileio_available else pathlib.Path(model_dir)