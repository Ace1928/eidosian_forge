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
def pdf_edit_text_field_value_outparams_fn(widget, value, change):
    """
    Class-aware helper for out-params of pdf_edit_text_field_value() [pdf_edit_text_field_value()].
    """
    ret, selStart, selEnd, newvalue = ll_pdf_edit_text_field_value(widget.m_internal, value, change)
    return (ret, selStart, selEnd, newvalue)