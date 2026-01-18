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
def ll_fz_new_document_writer_of_size(size, begin_page, end_page, close, drop):
    """
    Low-level wrapper for `::fz_new_document_writer_of_size()`.
    Internal function to allocate a
    block for a derived document_writer structure, with the base
    structure's function pointers populated correctly, and the extra
    space zero initialised.
    """
    return _mupdf.ll_fz_new_document_writer_of_size(size, begin_page, end_page, close, drop)