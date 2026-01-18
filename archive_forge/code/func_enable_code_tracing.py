from collections import namedtuple
import dis
import os
import re
import sys
from _pydev_bundle._pydev_saved_modules import threading
from types import CodeType, FrameType
from typing import Dict, Optional, Tuple, Any
from os.path import basename, splitext
from _pydev_bundle import pydev_log
from _pydevd_bundle import pydevd_dont_trace
from _pydevd_bundle.pydevd_constants import (GlobalDebuggerHolder, ForkSafeLock,
from pydevd_file_utils import (NORM_PATHS_AND_BASE_CONTAINER,
from _pydevd_bundle.pydevd_trace_dispatch import should_stop_on_exception, handle_exception
from _pydevd_bundle.pydevd_constants import EXCEPTION_TYPE_HANDLED
from _pydevd_bundle.pydevd_trace_dispatch import is_unhandled_exception
from _pydevd_bundle.pydevd_breakpoints import stop_on_unhandled_exception
from _pydevd_bundle.pydevd_utils import get_clsname_for_code
from _pydevd_bundle.pydevd_additional_thread_info import set_additional_thread_info, any_thread_stepping, PyDBAdditionalThreadInfo
def enable_code_tracing(thread_ident: Optional[int], code, frame) -> bool:
    """
    Note: this must enable code tracing for the given code/frame.

    The frame can be from any thread!

    :return: Whether code tracing was added in this function to the given code.
    """
    py_db: object = GlobalDebuggerHolder.global_dbg
    if py_db is None or py_db.pydb_disposed:
        return False
    func_code_info: FuncCodeInfo = _get_func_code_info(code, frame)
    if func_code_info.always_skip_code:
        return False
    try:
        thread = threading._active.get(thread_ident)
        if thread is None:
            return False
        additional_info = set_additional_thread_info(thread)
    except:
        return False
    return _enable_code_tracing(py_db, additional_info, func_code_info, code, frame, False)