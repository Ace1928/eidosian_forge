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
def set_diagnostic_callback(description, printfn):
    if g_mupdf_trace_director:
        log(f'set_diagnostic_callback() description={description!r} printfn={printfn!r}')
    if printfn:
        ret = DiagnosticCallbackPython(description, printfn)
        return ret
    else:
        if g_mupdf_trace_director:
            log(f'Calling ll_fz_set_{description}_callback() with (None, None)')
        if description == 'error':
            ll_fz_set_error_callback(None, None)
        elif description == 'warning':
            ll_fz_set_warning_callback(None, None)
        else:
            assert 0, f'Unrecognised description={description!r}'
        return None