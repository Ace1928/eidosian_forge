from _pydevd_bundle.pydevd_constants import EXCEPTION_TYPE_USER_UNHANDLED, EXCEPTION_TYPE_UNHANDLED, \
from _pydev_bundle import pydev_log
import itertools
from typing import Any, Dict
def create_frames_list_from_traceback(trace_obj, frame, exc_type, exc_desc, exception_type=None):
    """
    :param trace_obj:
        This is the traceback from which the list should be created.

    :param frame:
        This is the first frame to be considered (i.e.: topmost frame). If None is passed, all
        the frames from the traceback are shown (so, None should be passed for unhandled exceptions).

    :param exception_type:
        If this is an unhandled exception or user unhandled exception, we'll not trim the stack to create from the passed
        frame, rather, we'll just mark the frame in the frames list.
    """
    lst = []
    tb = trace_obj
    if tb is not None and tb.tb_frame is not None:
        f = tb.tb_frame.f_back
        while f is not None:
            lst.insert(0, (f, f.f_lineno, None))
            f = f.f_back
    while tb is not None:
        lst.append((tb.tb_frame, tb.tb_lineno, _get_line_col_info_from_tb(tb)))
        tb = tb.tb_next
    frames_list = None
    for tb_frame, tb_lineno, line_col_info in reversed(lst):
        if frames_list is None and (frame is tb_frame or frame is None or exception_type == EXCEPTION_TYPE_USER_UNHANDLED):
            frames_list = FramesList()
        if frames_list is not None:
            frames_list.append(tb_frame)
            frames_list.frame_id_to_lineno[id(tb_frame)] = tb_lineno
            frames_list.frame_id_to_line_col_info[id(tb_frame)] = line_col_info
    if frames_list is None and frame is not None:
        pydev_log.info('create_frames_list_from_traceback did not find topmost frame in list.')
        frames_list = create_frames_list_from_frame(frame)
    frames_list.exc_type = exc_type
    frames_list.exc_desc = exc_desc
    frames_list.trace_obj = trace_obj
    if exception_type == EXCEPTION_TYPE_USER_UNHANDLED:
        frames_list.current_frame = frame
    elif exception_type == EXCEPTION_TYPE_UNHANDLED:
        if len(frames_list) > 0:
            frames_list.current_frame = frames_list.last_frame()
    curr = frames_list
    memo = set()
    memo.add(id(exc_desc))
    while True:
        chained = create_frames_list_from_exception_cause(None, None, None, curr.exc_desc, memo)
        if chained is None:
            break
        else:
            curr.chained_frames_list = chained
            curr = chained
    return frames_list