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
def fz_set_error_callback(printfn):
    global set_error_callback_s
    set_error_callback_s = set_diagnostic_callback('error', printfn)