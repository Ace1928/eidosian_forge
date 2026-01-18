from _pydevd_bundle.pydevd_constants import EXCEPTION_TYPE_USER_UNHANDLED, EXCEPTION_TYPE_UNHANDLED, \
from _pydev_bundle import pydev_log
import itertools
from typing import Any, Dict
def create_frames_list_from_exception_cause(trace_obj, frame, exc_type, exc_desc, memo):
    lst = []
    msg = '<Unknown context>'
    try:
        exc_cause = getattr(exc_desc, '__cause__', None)
        msg = _cause_message
    except Exception:
        exc_cause = None
    if exc_cause is None:
        try:
            exc_cause = getattr(exc_desc, '__context__', None)
            msg = _context_message
        except Exception:
            exc_cause = None
    if exc_cause is None or id(exc_cause) in memo:
        return None
    memo.add(id(exc_cause))
    tb = exc_cause.__traceback__
    frames_list = FramesList()
    frames_list.exc_type = type(exc_cause)
    frames_list.exc_desc = exc_cause
    frames_list.trace_obj = tb
    frames_list.exc_context_msg = msg
    while tb is not None:
        lst.append((_DummyFrameWrapper(tb.tb_frame, tb.tb_lineno, None), tb.tb_lineno, _get_line_col_info_from_tb(tb)))
        tb = tb.tb_next
    for tb_frame, tb_lineno, line_col_info in lst:
        frames_list.append(tb_frame)
        frames_list.frame_id_to_lineno[id(tb_frame)] = tb_lineno
        frames_list.frame_id_to_line_col_info[id(tb_frame)] = line_col_info
    return frames_list