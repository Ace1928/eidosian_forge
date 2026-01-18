import pickle
from _pydevd_bundle.pydevd_constants import get_frame, get_current_thread_id, \
from _pydevd_bundle.pydevd_xml import ExceptionOnEvaluate, get_type, var_to_xml
from _pydev_bundle import pydev_log
import functools
from _pydevd_bundle.pydevd_thread_lifecycle import resume_threads, mark_thread_suspended, suspend_all_threads
from _pydevd_bundle.pydevd_comm_constants import CMD_SET_BREAK
import sys  # @Reimport
from _pydev_bundle._pydev_saved_modules import threading
from _pydevd_bundle import pydevd_save_locals, pydevd_timeout, pydevd_constants
from _pydev_bundle.pydev_imports import Exec, execfile
from _pydevd_bundle.pydevd_utils import to_string
import inspect
from _pydevd_bundle.pydevd_daemon_thread import PyDBDaemonThread
from _pydevd_bundle.pydevd_save_locals import update_globals_and_locals
from functools import lru_cache
@silence_warnings_decorator
def getVariable(dbg, thread_id, frame_id, scope, locator):
    """
    returns the value of a variable

    :scope: can be BY_ID, EXPRESSION, GLOBAL, LOCAL, FRAME

    BY_ID means we'll traverse the list of all objects alive to get the object.

    :locator: after reaching the proper scope, we have to get the attributes until we find
            the proper location (i.e.: obj	attr1	attr2)

    :note: when BY_ID is used, the frame_id is considered the id of the object to find and
           not the frame (as we don't care about the frame in this case).
    """
    if scope == 'BY_ID':
        if thread_id != get_current_thread_id(threading.current_thread()):
            raise VariableError('getVariable: must execute on same thread')
        try:
            import gc
            objects = gc.get_objects()
        except:
            pass
        else:
            frame_id = int(frame_id)
            for var in objects:
                if id(var) == frame_id:
                    if locator is not None:
                        locator_parts = locator.split('\t')
                        for k in locator_parts:
                            _type, _type_name, resolver = get_type(var)
                            var = resolver.resolve(var, k)
                    return var
        sys.stderr.write('Unable to find object with id: %s\n' % (frame_id,))
        return None
    frame = dbg.find_frame(thread_id, frame_id)
    if frame is None:
        return {}
    if locator is not None:
        locator_parts = locator.split('\t')
    else:
        locator_parts = []
    for attr in locator_parts:
        attr.replace('@_@TAB_CHAR@_@', '\t')
    if scope == 'EXPRESSION':
        for count in range(len(locator_parts)):
            if count == 0:
                var = evaluate_expression(dbg, frame, locator_parts[count], False)
            else:
                _type, _type_name, resolver = get_type(var)
                var = resolver.resolve(var, locator_parts[count])
    else:
        if scope == 'GLOBAL':
            var = frame.f_globals
            del locator_parts[0]
        else:
            var = {}
            var.update(frame.f_globals)
            var.update(frame.f_locals)
        for k in locator_parts:
            _type, _type_name, resolver = get_type(var)
            var = resolver.resolve(var, k)
    return var