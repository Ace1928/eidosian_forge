from _pydev_bundle import pydev_log
import os
from _pydevd_bundle.pydevd_comm import CMD_SIGNATURE_CALL_TRACE, NetCommand
from _pydevd_bundle import pydevd_xml
from _pydevd_bundle.pydevd_utils import get_clsname_for_code
def send_signature_return_trace(dbg, frame, filename, return_value):
    if dbg.signature_factory and dbg.in_project_scope(frame):
        signature = dbg.signature_factory.create_signature(frame, filename, with_args=False)
        signature.return_type = get_type_of_value(return_value, recursive=True)
        dbg.writer.add_command(create_signature_message(signature))
        return True
    return False