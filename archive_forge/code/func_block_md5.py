import hashlib
import io
import json
import os
import platform
import random
import socket
import ssl
import threading
import time
import urllib.parse
from typing import (
import filelock
import urllib3
from blobfile import _xml as xml
def block_md5(f: Any) -> bytes:
    m = hashlib.md5()
    while True:
        block = f.read(CHUNK_SIZE)
        if block == b'':
            break
        m.update(block)
    return m.digest()