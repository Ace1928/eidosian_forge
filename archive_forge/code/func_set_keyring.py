import configparser
import os
import sys
import logging
import typing
from . import backend, credentials
from .util import platform_ as platform
from .backends import fail
def set_keyring(keyring: backend.KeyringBackend) -> None:
    """Set current keyring backend."""
    global _keyring_backend
    if not isinstance(keyring, backend.KeyringBackend):
        raise TypeError('The keyring must be an instance of KeyringBackend')
    _keyring_backend = keyring