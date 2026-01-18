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
def ll_fz_md5_final(state, digest):
    """
    Low-level wrapper for `::fz_md5_final()`.
    MD5 finalization. Ends an MD5 message-digest operation, writing
    the message digest and zeroizing the context.

    Never throws an exception.
    """
    return _mupdf.ll_fz_md5_final(state, digest)