from __future__ import annotations
from abc import (
import codecs
from collections import defaultdict
from collections.abc import (
import dataclasses
import functools
import gzip
from io import (
import mmap
import os
from pathlib import Path
import re
import tarfile
from typing import (
from urllib.parse import (
import warnings
import zipfile
from pandas._typing import (
from pandas.compat import (
from pandas.compat._optional import import_optional_dependency
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.generic import ABCMultiIndex
from pandas.core.shared_docs import _shared_docs
def infer_filename(self) -> str | None:
    """
        If an explicit archive_name is not given, we still want the file inside the zip
        file not to be named something.zip, because that causes confusion (GH39465).
        """
    if isinstance(self.buffer.filename, (os.PathLike, str)):
        filename = Path(self.buffer.filename)
        if filename.suffix == '.zip':
            return filename.with_suffix('').name
        return filename.name
    return None