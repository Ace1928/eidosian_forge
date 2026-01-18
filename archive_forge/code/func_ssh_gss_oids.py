import struct
import os
import sys
from paramiko.common import MSG_USERAUTH_REQUEST
from paramiko.ssh_exception import SSHException
from paramiko._version import __version_info__
def ssh_gss_oids(self, mode='client'):
    """
        This method returns a single OID, because we only support the
        Kerberos V5 mechanism.

        :param str mode: Client for client mode and server for server mode
        :return: A byte sequence containing the number of supported
                 OIDs, the length of the OID and the actual OID encoded with
                 DER
        :note: In server mode we just return the OID length and the DER encoded
               OID.
        """
    from pyasn1.type.univ import ObjectIdentifier
    from pyasn1.codec.der import encoder
    OIDs = self._make_uint32(1)
    krb5_OID = encoder.encode(ObjectIdentifier(self._krb5_mech))
    OID_len = self._make_uint32(len(krb5_OID))
    if mode == 'server':
        return OID_len + krb5_OID
    return OIDs + OID_len + krb5_OID