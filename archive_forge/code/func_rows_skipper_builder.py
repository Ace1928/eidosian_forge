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
def rows_skipper_builder(cls, f, quotechar, is_quoting, encoding=None, newline=None):
    """
        Build object for skipping passed number of lines.

        Parameters
        ----------
        f : file-like object
            File handle that should be used for offset movement.
        quotechar : bytes
            Indicate quote in a file.
        is_quoting : bool
            Whether or not to consider quotes.
        encoding : str, optional
            Encoding of `f`.
        newline : bytes, optional
            Byte or sequence of bytes indicating line endings.

        Returns
        -------
        object
            skipper object.
        """

    def skipper(n):
        if n == 0 or n is None:
            return 0
        else:
            return cls._read_rows(f, quotechar=quotechar, is_quoting=is_quoting, nrows=n, encoding=encoding, newline=newline)[1]
    return skipper