from _pydev_bundle import pydev_log
import os
from _pydevd_bundle.pydevd_comm import CMD_SIGNATURE_CALL_TRACE, NetCommand
from _pydevd_bundle import pydevd_xml
from _pydevd_bundle.pydevd_utils import get_clsname_for_code
def send_signature_call_trace(dbg, frame, filename):
    if dbg.signature_factory and dbg.in_project_scope(frame):
        signature = dbg.signature_factory.create_signature(frame, filename)
        if signature is not None:
            if dbg.signature_factory.cache is not None:
                if not dbg.signature_factory.cache.is_in_cache(signature):
                    dbg.signature_factory.cache.add(signature)
                    dbg.writer.add_command(create_signature_message(signature))
                    return True
                else:
                    return False
            else:
                dbg.writer.add_command(create_signature_message(signature))
                return True
    return False