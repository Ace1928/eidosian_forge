from __future__ import annotations
from collections.abc import (
import datetime
from functools import partial
from io import BytesIO
import os
from textwrap import fill
from typing import (
import warnings
import zipfile
from pandas._config import config
from pandas._libs import lib
from pandas._libs.parsers import STR_NA_VALUES
from pandas.compat._optional import (
from pandas.errors import EmptyDataError
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import check_dtype_backend
from pandas.core.dtypes.common import (
from pandas.core.frame import DataFrame
from pandas.core.shared_docs import _shared_docs
from pandas.util.version import Version
from pandas.io.common import (
from pandas.io.excel._util import (
from pandas.io.parsers import TextParser
from pandas.io.parsers.readers import validate_integer
@classmethod
def check_extension(cls, ext: str) -> Literal[True]:
    """
        checks that path's extension against the Writer's supported
        extensions.  If it isn't supported, raises UnsupportedFiletypeError.
        """
    if ext.startswith('.'):
        ext = ext[1:]
    if not any((ext in extension for extension in cls._supported_extensions)):
        raise ValueError(f"Invalid extension for engine '{cls.engine}': '{ext}'")
    return True