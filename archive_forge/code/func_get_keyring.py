import configparser
import os
import sys
import logging
import typing
from . import backend, credentials
from .util import platform_ as platform
from .backends import fail
def get_keyring() -> backend.KeyringBackend:
    """Get current keyring backend."""
    if _keyring_backend is None:
        init_backend()
    return typing.cast(backend.KeyringBackend, _keyring_backend)