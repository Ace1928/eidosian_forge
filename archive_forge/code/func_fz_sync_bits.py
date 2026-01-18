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
def fz_sync_bits(self):
    """
        Class-aware wrapper for `::fz_sync_bits()`.
        	Called after reading bits to tell the stream
        	that we are about to return to reading bytewise. Resyncs
        	the stream to whole byte boundaries.
        """
    return _mupdf.FzStream_fz_sync_bits(self)