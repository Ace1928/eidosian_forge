from __future__ import unicode_literals
import base64
import uuid
import xml.etree.ElementTree as ET
import xmltodict
from six import text_type
from winrm.transport import Transport
from winrm.exceptions import WinRMError, WinRMTransportError, WinRMOperationTimeoutError
def open_shell(self, i_stream='stdin', o_stream='stdout stderr', working_directory=None, env_vars=None, noprofile=False, codepage=437, lifetime=None, idle_timeout=None):
    """
        Create a Shell on the destination host
        @param string i_stream: Which input stream to open. Leave this alone
         unless you know what you're doing (default: stdin)
        @param string o_stream: Which output stream to open. Leave this alone
         unless you know what you're doing (default: stdout stderr)
        @param string working_directory: the directory to create the shell in
        @param dict env_vars: environment variables to set for the shell. For
         instance: {'PATH': '%PATH%;c:/Program Files (x86)/Git/bin/', 'CYGWIN':
          'nontsec codepage:utf8'}
        @returns The ShellId from the SOAP response. This is our open shell
         instance on the remote machine.
        @rtype string
        """
    req = {'env:Envelope': self._get_soap_header(resource_uri='http://schemas.microsoft.com/wbem/wsman/1/windows/shell/cmd', action='http://schemas.xmlsoap.org/ws/2004/09/transfer/Create')}
    header = req['env:Envelope']['env:Header']
    header['w:OptionSet'] = {'w:Option': [{'@Name': 'WINRS_NOPROFILE', '#text': str(noprofile).upper()}, {'@Name': 'WINRS_CODEPAGE', '#text': str(codepage)}]}
    shell = req['env:Envelope'].setdefault('env:Body', {}).setdefault('rsp:Shell', {})
    shell['rsp:InputStreams'] = i_stream
    shell['rsp:OutputStreams'] = o_stream
    if working_directory:
        shell['rsp:WorkingDirectory'] = working_directory
    if idle_timeout:
        shell['rsp:IdleTimeOut'] = idle_timeout
    if env_vars:
        env = shell.setdefault('rsp:Environment', {}).setdefault('rsp:Variable', [])
        for key, value in env_vars.items():
            env.append({'@Name': key, '#text': value})
    res = self.send_message(xmltodict.unparse(req))
    root = ET.fromstring(res)
    return next((node for node in root.findall('.//*') if node.get('Name') == 'ShellId')).text