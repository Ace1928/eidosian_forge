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
def ll_fz_runeptr(str, idx):
    """
    Low-level wrapper for `::fz_runeptr()`.
    Obtain a pointer to the char representing the rune
    at a given index.

    str: Pointer to beginning of a string.

    idx: Index of a rune to return a char pointer to.

    Returns a pointer to the char where the desired rune starts,
    or NULL if the string ends before the index is reached.
    """
    return _mupdf.ll_fz_runeptr(str, idx)