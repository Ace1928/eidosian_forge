import linecache
import os.path
import re
from _pydev_bundle import pydev_log
from _pydevd_bundle import pydevd_dont_trace
from _pydevd_bundle.pydevd_constants import (RETURN_VALUES_DICT, NO_FTRACE,
from _pydevd_bundle.pydevd_frame_utils import add_exception_to_frame, just_raised, remove_exception_from_frame, ignore_exception_trace
from _pydevd_bundle.pydevd_utils import get_clsname_for_code
from pydevd_file_utils import get_abs_path_real_path_and_base_from_frame
from _pydevd_bundle.pydevd_comm_constants import constant_to_str, CMD_SET_FUNCTION_BREAK
import sys
import dis
def should_stop_on_exception(py_db, info, frame, thread, arg, prev_user_uncaught_exc_info):
    should_stop = False
    maybe_user_uncaught_exc_info = prev_user_uncaught_exc_info
    if info.pydev_state != 2:
        exception, value, trace = arg
        if trace is not None and hasattr(trace, 'tb_next'):
            should_stop = False
            exception_breakpoint = None
            try:
                if py_db.plugin is not None:
                    result = py_db.plugin.exception_break(py_db, frame, thread, arg)
                    if result:
                        should_stop, frame = result
            except:
                pydev_log.exception()
            if not should_stop:
                if exception == SystemExit and py_db.ignore_system_exit_code(value):
                    pass
                elif exception in (GeneratorExit, StopIteration, StopAsyncIteration):
                    pass
                elif ignore_exception_trace(trace):
                    pass
                else:
                    was_just_raised = trace.tb_next is None
                    check_excs = []
                    exc_break_user = py_db.get_exception_breakpoint(exception, py_db.break_on_user_uncaught_exceptions)
                    if exc_break_user is not None:
                        check_excs.append((exc_break_user, True))
                    exc_break_caught = py_db.get_exception_breakpoint(exception, py_db.break_on_caught_exceptions)
                    if exc_break_caught is not None:
                        check_excs.append((exc_break_caught, False))
                    for exc_break, is_user_uncaught in check_excs:
                        should_stop = True
                        if py_db.exclude_exception_by_filter(exc_break, trace):
                            pydev_log.debug('Ignore exception %s in library %s -- (%s)' % (exception, frame.f_code.co_filename, frame.f_code.co_name))
                            should_stop = False
                        elif exc_break.condition is not None and (not py_db.handle_breakpoint_condition(info, exc_break, frame)):
                            should_stop = False
                        elif is_user_uncaught:
                            should_stop = False
                            if not py_db.apply_files_filter(frame, frame.f_code.co_filename, True) and (frame.f_back is None or py_db.apply_files_filter(frame.f_back, frame.f_back.f_code.co_filename, True)):
                                exc_info = prev_user_uncaught_exc_info
                                if not exc_info:
                                    exc_info = (arg, frame.f_lineno, set([frame.f_lineno]))
                                else:
                                    lines = exc_info[2]
                                    lines.add(frame.f_lineno)
                                    exc_info = (arg, frame.f_lineno, lines)
                                maybe_user_uncaught_exc_info = exc_info
                        elif exc_break.notify_on_first_raise_only and py_db.skip_on_exceptions_thrown_in_same_context and (not was_just_raised) and (not just_raised(trace.tb_next)):
                            should_stop = False
                        elif exc_break.notify_on_first_raise_only and (not py_db.skip_on_exceptions_thrown_in_same_context) and (not was_just_raised):
                            should_stop = False
                        elif was_just_raised and py_db.skip_on_exceptions_thrown_in_same_context:
                            should_stop = False
                        if should_stop:
                            exception_breakpoint = exc_break
                            try:
                                info.pydev_message = exc_break.qname
                            except:
                                info.pydev_message = exc_break.qname.encode('utf-8')
                            break
            if should_stop:
                add_exception_to_frame(frame, (exception, value, trace))
                if exception_breakpoint is not None and exception_breakpoint.expression is not None:
                    py_db.handle_breakpoint_expression(exception_breakpoint, info, frame)
    return (should_stop, frame, maybe_user_uncaught_exc_info)