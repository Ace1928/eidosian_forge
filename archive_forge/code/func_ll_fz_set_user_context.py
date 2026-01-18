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
def ll_fz_set_user_context(user):
    """
    Low-level wrapper for `::fz_set_user_context()`.
    Set the user field in the context.

    NULL initially, this field can be set to any opaque value
    required by the user. It is copied on clones.
    """
    return _mupdf.ll_fz_set_user_context(user)