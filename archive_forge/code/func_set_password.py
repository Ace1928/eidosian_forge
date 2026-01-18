import platform
import os
import warnings
import functools
from ...backend import KeyringBackend
from ...errors import PasswordSetError
from ...errors import PasswordDeleteError
from ...errors import KeyringLocked
from ...errors import KeyringError
from ..._compat import properties
@warn_keychain
def set_password(self, service, username, password):
    if username is None:
        username = ''
    try:
        api.set_generic_password(self.keychain, service, username, password)
    except api.KeychainDenied as e:
        raise KeyringLocked("Can't store password on keychain: {}".format(e))
    except api.Error as e:
        raise PasswordSetError("Can't store password on keychain: {}".format(e))