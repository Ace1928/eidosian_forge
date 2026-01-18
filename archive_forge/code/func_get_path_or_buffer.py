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
def get_path_or_buffer(cls, filepath_or_buffer):
    """
        Extract path from `filepath_or_buffer`.

        Parameters
        ----------
        filepath_or_buffer : str, path object or file-like object
            `filepath_or_buffer` parameter of `read_csv` function.

        Returns
        -------
        str or path object
            verified `filepath_or_buffer` parameter.

        Notes
        -----
        Given a buffer, try and extract the filepath from it so that we can
        use it without having to fall back to pandas and share file objects between
        workers. Given a filepath, return it immediately.
        """
    if hasattr(filepath_or_buffer, 'name') and hasattr(filepath_or_buffer, 'seekable') and filepath_or_buffer.seekable() and (filepath_or_buffer.tell() == 0):
        buffer_filepath = filepath_or_buffer.name
        if cls.file_exists(buffer_filepath):
            warnings.warn('For performance reasons, the filepath will be ' + 'used in place of the file handle passed in ' + 'to load the data')
            return cls.get_path(buffer_filepath)
    return filepath_or_buffer