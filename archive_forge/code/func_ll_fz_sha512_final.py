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
def ll_fz_sha512_final(state, digest):
    """
    Low-level wrapper for `::fz_sha512_final()`.
    SHA512 finalization. Ends an SHA512 message-digest operation,
    writing the message digest and zeroizing the context.

    Never throws an exception.
    """
    return _mupdf.ll_fz_sha512_final(state, digest)