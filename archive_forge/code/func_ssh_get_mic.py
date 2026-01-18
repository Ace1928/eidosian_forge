import struct
import os
import sys
from paramiko.common import MSG_USERAUTH_REQUEST
from paramiko.ssh_exception import SSHException
from paramiko._version import __version_info__
def ssh_get_mic(self, session_id, gss_kex=False):
    """
        Create the MIC token for a SSH2 message.

        :param str session_id: The SSH session ID
        :param bool gss_kex: Generate the MIC for Key Exchange with SSPI or not
        :return: gssapi-with-mic:
                 Returns the MIC token from SSPI for the message we created
                 with ``_ssh_build_mic``.
                 gssapi-keyex:
                 Returns the MIC token from SSPI with the SSH session ID as
                 message.
        """
    self._session_id = session_id
    if not gss_kex:
        mic_field = self._ssh_build_mic(self._session_id, self._username, self._service, self._auth_method)
        mic_token = self._gss_ctxt.sign(mic_field)
    else:
        mic_token = self._gss_srv_ctxt.sign(self._session_id)
    return mic_token