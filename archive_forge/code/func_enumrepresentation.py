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
def enumrepresentation(options):
    if options['format'] == 'md5-hex':
        options['format'] = keys.FingerprintFormats.MD5_HEX
        return options
    elif options['format'] == 'sha256-base64':
        options['format'] = keys.FingerprintFormats.SHA256_BASE64
        return options
    else:
        raise keys.BadFingerPrintFormat(f'Unsupported fingerprint format: {options['format']}')