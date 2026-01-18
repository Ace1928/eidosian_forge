import time
from _pydev_bundle._pydev_filesystem_encoding import getfilesystemencoding
from _pydev_bundle._pydev_saved_modules import threading
from _pydevd_bundle import pydevd_xml
from _pydevd_bundle.pydevd_constants import GlobalDebuggerHolder
from _pydevd_bundle.pydevd_constants import get_thread_id
from _pydevd_bundle.pydevd_net_command import NetCommand
from _pydevd_bundle.pydevd_concurrency_analyser.pydevd_thread_wrappers import ObjectWrapper, wrap_attr
import pydevd_file_utils
from _pydev_bundle import pydev_log
import sys
from urllib.parse import quote
def send_concurrency_message(event_class, time, name, thread_id, type, event, file, line, frame, lock_id=0, parent=None):
    dbg = GlobalDebuggerHolder.global_dbg
    if dbg is None:
        return
    cmdTextList = ['<xml>']
    cmdTextList.append('<' + event_class)
    cmdTextList.append(' time="%s"' % pydevd_xml.make_valid_xml_value(str(time)))
    cmdTextList.append(' name="%s"' % pydevd_xml.make_valid_xml_value(name))
    cmdTextList.append(' thread_id="%s"' % pydevd_xml.make_valid_xml_value(thread_id))
    cmdTextList.append(' type="%s"' % pydevd_xml.make_valid_xml_value(type))
    if type == 'lock':
        cmdTextList.append(' lock_id="%s"' % pydevd_xml.make_valid_xml_value(str(lock_id)))
    if parent is not None:
        cmdTextList.append(' parent="%s"' % pydevd_xml.make_valid_xml_value(parent))
    cmdTextList.append(' event="%s"' % pydevd_xml.make_valid_xml_value(event))
    cmdTextList.append(' file="%s"' % pydevd_xml.make_valid_xml_value(file))
    cmdTextList.append(' line="%s"' % pydevd_xml.make_valid_xml_value(str(line)))
    cmdTextList.append('></' + event_class + '>')
    cmdTextList += get_text_list_for_frame(frame)
    cmdTextList.append('</xml>')
    text = ''.join(cmdTextList)
    if dbg.writer is not None:
        dbg.writer.add_command(NetCommand(145, 0, text))