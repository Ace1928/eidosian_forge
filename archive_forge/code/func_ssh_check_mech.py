import struct
import os
import sys
from paramiko.common import MSG_USERAUTH_REQUEST
from paramiko.ssh_exception import SSHException
from paramiko._version import __version_info__
def ssh_check_mech(self, desired_mech):
    """
        Check if the given OID is the Kerberos V5 OID (server mode).

        :param str desired_mech: The desired GSS-API mechanism of the client
        :return: ``True`` if the given OID is supported, otherwise C{False}
        """
    from pyasn1.codec.der import decoder
    mech, __ = decoder.decode(desired_mech)
    if mech.__str__() != self._krb5_mech:
        return False
    return True