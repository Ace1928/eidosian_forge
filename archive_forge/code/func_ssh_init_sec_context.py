import struct
import os
import sys
from paramiko.common import MSG_USERAUTH_REQUEST
from paramiko.ssh_exception import SSHException
from paramiko._version import __version_info__
def ssh_init_sec_context(self, target, desired_mech=None, username=None, recv_token=None):
    """
        Initialize a SSPI context.

        :param str username: The name of the user who attempts to login
        :param str target: The FQDN of the target to connect to
        :param str desired_mech: The negotiated SSPI mechanism
                                 ("pseudo negotiated" mechanism, because we
                                 support just the krb5 mechanism :-))
        :param recv_token: The SSPI token received from the Server
        :raises:
            `.SSHException` -- Is raised if the desired mechanism of the client
            is not supported
        :return: A ``String`` if the SSPI has returned a token or ``None`` if
                 no token was returned
        """
    from pyasn1.codec.der import decoder
    self._username = username
    self._gss_host = target
    error = 0
    targ_name = 'host/' + self._gss_host
    if desired_mech is not None:
        mech, __ = decoder.decode(desired_mech)
        if mech.__str__() != self._krb5_mech:
            raise SSHException('Unsupported mechanism OID.')
    try:
        if recv_token is None:
            self._gss_ctxt = sspi.ClientAuth('Kerberos', scflags=self._gss_flags, targetspn=targ_name)
        error, token = self._gss_ctxt.authorize(recv_token)
        token = token[0].Buffer
    except pywintypes.error as e:
        e.strerror += ', Target: {}'.format(self._gss_host)
        raise
    if error == 0:
        '\n            if the status is GSS_COMPLETE (error = 0) the context is fully\n            established an we can set _gss_ctxt_status to True.\n            '
        self._gss_ctxt_status = True
        token = None
        '\n            You won\'t get another token if the context is fully established,\n            so i set token to None instead of ""\n            '
    return token