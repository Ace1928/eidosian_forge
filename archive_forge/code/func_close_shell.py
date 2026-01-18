from __future__ import unicode_literals
import base64
import uuid
import xml.etree.ElementTree as ET
import xmltodict
from six import text_type
from winrm.transport import Transport
from winrm.exceptions import WinRMError, WinRMTransportError, WinRMOperationTimeoutError
def close_shell(self, shell_id, close_session=True):
    """
        Close the shell
        @param string shell_id: The shell id on the remote machine.
         See #open_shell
        @param bool close_session: If we want to close the requests's session.
         Allows to completely close all TCP connections to the server.
        @returns This should have more error checking but it just returns true
         for now.
        @rtype bool
        """
    try:
        message_id = uuid.uuid4()
        req = {'env:Envelope': self._get_soap_header(resource_uri='http://schemas.microsoft.com/wbem/wsman/1/windows/shell/cmd', action='http://schemas.xmlsoap.org/ws/2004/09/transfer/Delete', shell_id=shell_id, message_id=message_id)}
        req['env:Envelope'].setdefault('env:Body', {})
        res = self.send_message(xmltodict.unparse(req))
        root = ET.fromstring(res)
        relates_to = next((node for node in root.findall('.//*') if node.tag.endswith('RelatesTo'))).text
    finally:
        if close_session:
            self.transport.close_session()
    assert uuid.UUID(relates_to.replace('uuid:', '')) == message_id