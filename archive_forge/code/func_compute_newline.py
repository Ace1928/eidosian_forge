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
def compute_newline(cls, file_like, encoding, quotechar):
    """
        Compute byte or sequence of bytes indicating line endings.

        Parameters
        ----------
        file_like : file-like object
            File handle that should be used for line endings computing.
        encoding : str
            Encoding of `file_like`.
        quotechar : str
            Quotechar used for parsing `file-like`.

        Returns
        -------
        bytes
            line endings
        """
    newline = None
    if encoding is None:
        return (newline, quotechar.encode('UTF-8'))
    quotechar = quotechar.encode(encoding)
    encoding = encoding.replace('-', '_')
    if 'utf' in encoding and '8' not in encoding or encoding == 'unicode_escape' or encoding == 'utf_8_sig':
        file_like.readline()
        newline = file_like.newlines.encode(encoding)
        boms = ()
        if encoding == 'utf_8_sig':
            boms = (codecs.BOM_UTF8,)
        elif '16' in encoding:
            boms = (codecs.BOM_UTF16_BE, codecs.BOM_UTF16_LE)
        elif '32' in encoding:
            boms = (codecs.BOM_UTF32_BE, codecs.BOM_UTF32_LE)
        for bom in boms:
            if newline.startswith(bom):
                bom_len = len(bom)
                newline = newline[bom_len:]
                quotechar = quotechar[bom_len:]
                break
    return (newline, quotechar)