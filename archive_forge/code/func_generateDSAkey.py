from __future__ import annotations
import getpass
import os
import platform
import socket
import sys
from collections.abc import Callable
from functools import wraps
from importlib import reload
from typing import Any, Dict, Optional
from twisted.conch.ssh import keys
from twisted.python import failure, filepath, log, usage
@_keyGenerator('dsa')
def generateDSAkey(options):
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives.asymmetric import dsa
    if not options['bits']:
        options['bits'] = 1024
    keyPrimitive = dsa.generate_private_key(key_size=int(options['bits']), backend=default_backend())
    key = keys.Key(keyPrimitive)
    _saveKey(key, options)