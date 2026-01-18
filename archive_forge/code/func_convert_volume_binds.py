import base64
import collections
import json
import os
import os.path
import shlex
import string
from datetime import datetime
from packaging.version import Version
from .. import errors
from ..constants import DEFAULT_HTTP_HOST
from ..constants import DEFAULT_UNIX_SOCKET
from ..constants import DEFAULT_NPIPE
from ..constants import BYTE_UNITS
from ..tls import TLSConfig
from urllib.parse import urlparse, urlunparse
def convert_volume_binds(binds):
    if isinstance(binds, list):
        return binds
    result = []
    for k, v in binds.items():
        if isinstance(k, bytes):
            k = k.decode('utf-8')
        if isinstance(v, dict):
            if 'ro' in v and 'mode' in v:
                raise ValueError('Binding cannot contain both "ro" and "mode": {}'.format(repr(v)))
            bind = v['bind']
            if isinstance(bind, bytes):
                bind = bind.decode('utf-8')
            if 'ro' in v:
                mode = 'ro' if v['ro'] else 'rw'
            elif 'mode' in v:
                mode = v['mode']
            else:
                mode = 'rw'
            result.append(f'{k}:{bind}:{mode}')
        else:
            if isinstance(v, bytes):
                v = v.decode('utf-8')
            result.append(f'{k}:{v}:rw')
    return result