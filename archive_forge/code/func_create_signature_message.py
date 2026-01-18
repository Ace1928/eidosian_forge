from _pydev_bundle import pydev_log
import os
from _pydevd_bundle.pydevd_comm import CMD_SIGNATURE_CALL_TRACE, NetCommand
from _pydevd_bundle import pydevd_xml
from _pydevd_bundle.pydevd_utils import get_clsname_for_code
def create_signature_message(signature):
    cmdTextList = ['<xml>']
    cmdTextList.append('<call_signature file="%s" name="%s">' % (pydevd_xml.make_valid_xml_value(signature.file), pydevd_xml.make_valid_xml_value(signature.name)))
    for arg in signature.args:
        cmdTextList.append('<arg name="%s" type="%s"></arg>' % (pydevd_xml.make_valid_xml_value(arg[0]), pydevd_xml.make_valid_xml_value(arg[1])))
    if signature.return_type is not None:
        cmdTextList.append('<return type="%s"></return>' % pydevd_xml.make_valid_xml_value(signature.return_type))
    cmdTextList.append('</call_signature></xml>')
    cmdText = ''.join(cmdTextList)
    return NetCommand(CMD_SIGNATURE_CALL_TRACE, 0, cmdText)