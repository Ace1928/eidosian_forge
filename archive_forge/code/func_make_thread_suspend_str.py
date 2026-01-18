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
def make_thread_suspend_str(self, py_db, thread_id, frames_list, stop_reason=None, message=None, suspend_type='trace'):
    """
        :return tuple(str,str):
            Returns tuple(thread_suspended_str, thread_stack_str).

            i.e.:
            (
                '''
                    <xml>
                        <thread id="id" stop_reason="reason">
                            <frame id="id" name="functionName " file="file" line="line">
                            </frame>
                        </thread>
                    </xml>
                '''
                ,
                '''
                <frame id="id" name="functionName " file="file" line="line">
                </frame>
                '''
            )
        """
    assert frames_list.__class__ == FramesList
    make_valid_xml_value = pydevd_xml.make_valid_xml_value
    cmd_text_list = []
    append = cmd_text_list.append
    cmd_text_list.append('<xml>')
    if message:
        message = make_valid_xml_value(message)
    append('<thread id="%s"' % (thread_id,))
    if stop_reason is not None:
        append(' stop_reason="%s"' % (stop_reason,))
    if message is not None:
        append(' message="%s"' % (message,))
    if suspend_type is not None:
        append(' suspend_type="%s"' % (suspend_type,))
    append('>')
    thread_stack_str = self.make_thread_stack_str(py_db, frames_list)
    append(thread_stack_str)
    append('</thread></xml>')
    return (''.join(cmd_text_list), thread_stack_str)