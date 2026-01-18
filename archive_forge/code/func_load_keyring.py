import configparser
import os
import sys
import logging
import typing
from . import backend, credentials
from .util import platform_ as platform
from .backends import fail
def load_keyring(keyring_name: str) -> backend.KeyringBackend:
    """
    Load the specified keyring by name (a fully-qualified name to the
    keyring, such as 'keyring.backends.file.PlaintextKeyring')
    """
    class_ = _load_keyring_class(keyring_name)
    class_.priority
    return class_()