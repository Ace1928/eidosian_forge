import sys
import os
import contextlib
from ..backend import KeyringBackend
from ..credentials import SimpleCredential
from ..errors import PasswordDeleteError
from ..errors import PasswordSetError, InitError, KeyringLocked
from .._compat import properties
def get_credential(self, service, username):
    """Gets the first username and password for a service.
        Returns a Credential instance

        The username can be omitted, but if there is one, it will forward to
        get_password.
        Otherwise, it will return the first username and password combo that it finds.
        """
    if username is not None:
        return super().get_credential(service, username)
    if not self.connected(service):
        raise KeyringLocked('Failed to unlock the keyring!')
    for username in self.iface.entryList(self.handle, service, self.appid):
        password = self.iface.readPassword(self.handle, service, username, self.appid)
        return SimpleCredential(str(username), str(password))