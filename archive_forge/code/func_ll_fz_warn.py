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
def ll_fz_warn(text):
    assert isinstance(text, str), f'text={text!r} str={str!r}'
    text = text.replace('%', '%%')
    return ll_fz_warn_original(text)