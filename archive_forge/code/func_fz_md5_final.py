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
def fz_md5_final(self, digest):
    """
        We use default copy constructor and operator=.  Class-aware wrapper for `::fz_md5_final()`.
        	MD5 finalization. Ends an MD5 message-digest operation, writing
        	the message digest and zeroizing the context.

        	Never throws an exception.
        """
    return _mupdf.FzMd5_fz_md5_final(self, digest)