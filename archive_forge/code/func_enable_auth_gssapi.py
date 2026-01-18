import threading
from paramiko import util
from paramiko.common import (
def enable_auth_gssapi(self):
    """
        Overwrite this function in your SSH server to enable GSSAPI
        authentication.
        The default implementation always returns false.

        :returns bool: Whether GSSAPI authentication is enabled.
        :see: `.ssh_gss`
        """
    UseGSSAPI = False
    return UseGSSAPI