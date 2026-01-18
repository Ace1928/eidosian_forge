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
def resolve_compound_variable_fields(dbg, thread_id, frame_id, scope, attrs):
    """
    Resolve compound variable in debugger scopes by its name and attributes

    :param thread_id: id of the variable's thread
    :param frame_id: id of the variable's frame
    :param scope: can be BY_ID, EXPRESSION, GLOBAL, LOCAL, FRAME
    :param attrs: after reaching the proper scope, we have to get the attributes until we find
            the proper location (i.e.: obj	attr1	attr2)
    :return: a dictionary of variables's fields
    """
    var = getVariable(dbg, thread_id, frame_id, scope, attrs)
    try:
        _type, type_name, resolver = get_type(var)
        return (type_name, resolver.get_dictionary(var))
    except:
        pydev_log.exception('Error evaluating: thread_id: %s\nframe_id: %s\nscope: %s\nattrs: %s.', thread_id, frame_id, scope, attrs)