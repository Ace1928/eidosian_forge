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
def make_list_threads_message(self, py_db, seq):
    """ returns thread listing as XML """
    try:
        threads = get_non_pydevd_threads()
        cmd_text = ['<xml>']
        append = cmd_text.append
        for thread in threads:
            if is_thread_alive(thread):
                append(self._thread_to_xml(thread))
        for thread_id, thread_name in list(self._additional_thread_id_to_thread_name.items()):
            name = pydevd_xml.make_valid_xml_value(thread_name)
            append('<thread name="%s" id="%s" />' % (quote(name), thread_id))
        append('</xml>')
        return NetCommand(CMD_RETURN, seq, ''.join(cmd_text))
    except:
        return self.make_error_message(seq, get_exception_traceback_str())