import os
import fsspec
import numpy as np
from pandas.io.common import is_fsspec_url, is_url
from modin.config import AsyncReadMode
from modin.logging import ClassLogger
from modin.utils import ModinAssumptionError
@classmethod
def file_size(cls, f):
    """
        Get the size of file associated with file handle `f`.

        Parameters
        ----------
        f : file-like object
            File-like object, that should be used to get file size.

        Returns
        -------
        int
            File size in bytes.
        """
    cur_pos = f.tell()
    f.seek(0, os.SEEK_END)
    size = f.tell()
    f.seek(cur_pos, os.SEEK_SET)
    return size