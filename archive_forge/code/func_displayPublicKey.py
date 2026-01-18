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
def displayPublicKey(options):
    filename = _getKeyOrDefault(options)
    try:
        key = keys.Key.fromFile(filename)
    except FileNotFoundError:
        sys.exit(f'{filename} could not be opened, please specify a file.')
    except keys.EncryptedKeyError:
        if not options.get('pass'):
            options['pass'] = getpass.getpass('Enter passphrase: ')
        key = keys.Key.fromFile(filename, passphrase=options['pass'])
    displayKey = key.public().toString('openssh').decode('ascii')
    print(displayKey)