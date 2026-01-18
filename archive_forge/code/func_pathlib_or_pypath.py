import codecs
import io
import os
import warnings
from csv import QUOTE_NONE
from typing import Callable, Optional, Sequence, Tuple, Union
import numpy as np
import pandas
import pandas._libs.lib as lib
from pandas.core.dtypes.common import is_list_like
from pandas.io.common import stringify_path
from modin.config import MinPartitionSize, NPartitions
from modin.core.io.file_dispatcher import FileDispatcher, OpenFile
from modin.core.io.text.utils import CustomNewlineIterator
from modin.core.storage_formats.pandas.utils import compute_chunksize
from modin.utils import _inherit_docstrings
@classmethod
def pathlib_or_pypath(cls, filepath_or_buffer):
    """
        Check if `filepath_or_buffer` is instance of `py.path.local` or `pathlib.Path`.

        Parameters
        ----------
        filepath_or_buffer : str, path object or file-like object
            `filepath_or_buffer` parameter of `read_csv` function.

        Returns
        -------
        bool
            Whether or not `filepath_or_buffer` is instance of `py.path.local`
            or `pathlib.Path`.
        """
    try:
        import py
        if isinstance(filepath_or_buffer, py.path.local):
            return True
    except ImportError:
        pass
    try:
        import pathlib
        if isinstance(filepath_or_buffer, pathlib.Path):
            return True
    except ImportError:
        pass
    return False