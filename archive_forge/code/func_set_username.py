import struct
import os
import sys
from paramiko.common import MSG_USERAUTH_REQUEST
from paramiko.ssh_exception import SSHException
from paramiko._version import __version_info__
def set_username(self, username):
    """
        Setter for C{username}. If GSS-API Key Exchange is performed, the
        username is not set by C{ssh_init_sec_context}.

        :param str username: The name of the user who attempts to login
        """
    self._username = username