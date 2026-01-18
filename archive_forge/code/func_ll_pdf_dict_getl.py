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
def ll_pdf_dict_getl(obj, *tail):
    """
    Python implementation of ll_pdf_dict_getl(), because SWIG
    doesn't handle variadic args. Each item in `tail` should be
    `mupdf.pdf_obj`.
    """
    for key in tail:
        if not obj:
            break
        obj = ll_pdf_dict_get(obj, key)
    assert isinstance(obj, pdf_obj)
    return obj