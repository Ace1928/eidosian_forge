from __future__ import absolute_import, print_function, division
import io
from petl.compat import next, PY2, text_type
from petl.util.base import Table, asdict
from petl.io.base import getcodec
from petl.io.sources import read_source_from_arg, write_source_from_arg
def teetext(table, source=None, encoding=None, errors='strict', template=None, prologue=None, epilogue=None):
    """
    Return a table that writes rows to a text file as they are iterated over.

    """
    assert template is not None, 'template is required'
    return TeeTextView(table, source=source, encoding=encoding, errors=errors, template=template, prologue=prologue, epilogue=epilogue)