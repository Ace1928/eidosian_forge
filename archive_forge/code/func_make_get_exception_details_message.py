import json
from _pydev_bundle.pydev_is_thread_alive import is_thread_alive
from _pydev_bundle._pydev_saved_modules import thread
from _pydevd_bundle import pydevd_xml, pydevd_frame_utils, pydevd_constants, pydevd_utils
from _pydevd_bundle.pydevd_comm_constants import (
from _pydevd_bundle.pydevd_constants import (DebugInfoHolder, get_thread_id,
from _pydevd_bundle.pydevd_net_command import NetCommand, NULL_NET_COMMAND, NULL_EXIT_COMMAND
from _pydevd_bundle.pydevd_utils import quote_smart as quote, get_non_pydevd_threads
from pydevd_file_utils import get_abs_path_real_path_and_base_from_frame
import pydevd_file_utils
from pydevd_tracing import get_exception_traceback_str
from _pydev_bundle._pydev_completer import completions_to_xml
from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_frame_utils import FramesList
from io import StringIO
def make_get_exception_details_message(self, py_db, seq, thread_id, topmost_frame):
    """Returns exception details as XML """
    try:
        cmd_text = ['<xml><thread id="%s" ' % (thread_id,)]
        if topmost_frame is not None:
            try:
                frame = topmost_frame
                topmost_frame = None
                while frame is not None:
                    if frame.f_code.co_name == 'do_wait_suspend' and frame.f_code.co_filename.endswith('pydevd.py'):
                        arg = frame.f_locals.get('arg', None)
                        if arg is not None:
                            exc_type, exc_desc, _thread_suspend_str, thread_stack_str = self._make_send_curr_exception_trace_str(py_db, thread_id, *arg)
                            cmd_text.append('exc_type="%s" ' % (exc_type,))
                            cmd_text.append('exc_desc="%s" ' % (exc_desc,))
                            cmd_text.append('>')
                            cmd_text.append(thread_stack_str)
                            break
                    frame = frame.f_back
                else:
                    cmd_text.append('>')
            finally:
                frame = None
        cmd_text.append('</thread></xml>')
        return NetCommand(CMD_GET_EXCEPTION_DETAILS, seq, ''.join(cmd_text))
    except:
        return self.make_error_message(seq, get_exception_traceback_str())