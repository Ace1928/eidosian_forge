import struct
import os
import sys
from paramiko.common import MSG_USERAUTH_REQUEST
from paramiko.ssh_exception import SSHException
from paramiko._version import __version_info__
@property
def credentials_delegated(self):
    """
        Checks if credentials are delegated (server mode).

        :return: ``True`` if credentials are delegated, otherwise ``False``
        """
    return self._gss_flags & sspicon.ISC_REQ_DELEGATE and (self._gss_srv_ctxt_status or self._gss_flags)