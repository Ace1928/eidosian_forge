import struct
import os
import sys
from paramiko.common import MSG_USERAUTH_REQUEST
from paramiko.ssh_exception import SSHException
from paramiko._version import __version_info__
def set_service(self, service):
    """
        This is just a setter to use a non default service.
        I added this method, because RFC 4462 doesn't specify "ssh-connection"
        as the only service value.

        :param str service: The desired SSH service
        """
    if service.find('ssh-'):
        self._service = service