import struct
import os
import sys
from paramiko.common import MSG_USERAUTH_REQUEST
from paramiko.ssh_exception import SSHException
from paramiko._version import __version_info__
def save_client_creds(self, client_token):
    """
        Save the Client token in a file. This is used by the SSH server
        to store the client credentails if credentials are delegated
        (server mode).

        :param str client_token: The SSPI token received form the client
        :raises:
            ``NotImplementedError`` -- Credential delegation is currently not
            supported in server mode
        """
    raise NotImplementedError