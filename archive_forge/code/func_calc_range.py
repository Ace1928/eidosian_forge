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
def calc_range(start: Optional[int]=None, end: Optional[int]=None) -> str:
    if start is not None and end is not None:
        return f'bytes={start}-{end - 1}'
    elif start is not None:
        return f'bytes={start}-'
    elif end is not None:
        if end > 0:
            return f'bytes=0-{end - 1}'
        else:
            return f'bytes=-{-int(end)}'
    else:
        raise Error('Invalid range')