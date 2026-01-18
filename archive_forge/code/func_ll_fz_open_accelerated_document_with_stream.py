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
def ll_fz_open_accelerated_document_with_stream(magic, stream, accel):
    """
    Low-level wrapper for `::fz_open_accelerated_document_with_stream()`.
    Open a document using the specified stream object rather than
    opening a file on disk.

    magic: a string used to detect document type; either a file name
    or mime-type.

    stream: a stream of the document contents.

    accel: NULL, or a stream of the 'accelerator' contents for this document.

    NOTE: The caller retains ownership of 'stream' and 'accel' - the document will
    take its own references if required.
    """
    return _mupdf.ll_fz_open_accelerated_document_with_stream(magic, stream, accel)