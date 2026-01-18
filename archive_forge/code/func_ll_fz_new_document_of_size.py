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
def ll_fz_new_document_of_size(size):
    """
    Low-level wrapper for `::fz_new_document_of_size()`.
    New documents are typically created by calls like
    foo_new_document(fz_context *ctx, ...). These work by
    deriving a new document type from fz_document, for instance:
    typedef struct { fz_document base; ...extras... } foo_document;
    These are allocated by calling
    fz_new_derived_document(ctx, foo_document)
    """
    return _mupdf.ll_fz_new_document_of_size(size)