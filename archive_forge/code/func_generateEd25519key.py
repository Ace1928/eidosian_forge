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
@_keyGenerator('ed25519')
def generateEd25519key(options):
    keyPrimitive = keys.Ed25519PrivateKey.generate()
    key = keys.Key(keyPrimitive)
    _saveKey(key, options)