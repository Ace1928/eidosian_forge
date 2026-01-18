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
def printFingerprint(options: Dict[Any, Any]) -> None:
    filename = _getKeyOrDefault(options)
    if os.path.exists(filename + '.pub'):
        filename += '.pub'
    options = enumrepresentation(options)
    try:
        key = keys.Key.fromFile(filename)
        print('%s %s %s' % (key.size(), key.fingerprint(options['format']), os.path.basename(filename)))
    except keys.BadKeyError:
        sys.exit('bad key')
    except FileNotFoundError:
        sys.exit(f'{filename} could not be opened, please specify a file.')