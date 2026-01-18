from sys import version_info as _swig_python_version_info
import weakref
import inspect
import os
import re
import sys
import traceback
import inspect
import io
import os
import sys
import traceback
import types
def fz_open_faxd(self, k, end_of_line, encoded_byte_align, columns, rows, end_of_block, black_is_1):
    """
        Class-aware wrapper for `::fz_open_faxd()`.
        	faxd filter performs FAX decoding of data read from
        	the chained filter.

        	k: see fax specification (fax default is 0).

        	end_of_line: whether we expect end of line markers (fax default
        	is 0).

        	encoded_byte_align: whether we align to bytes after each line
        	(fax default is 0).

        	columns: how many columns in the image (fax default is 1728).

        	rows: 0 for unspecified or the number of rows of data to expect.

        	end_of_block: whether we expect end of block markers (fax
        	default is 1).

        	black_is_1: determines the polarity of the image (fax default is
        	0).
        """
    return _mupdf.FzStream_fz_open_faxd(self, k, end_of_line, encoded_byte_align, columns, rows, end_of_block, black_is_1)