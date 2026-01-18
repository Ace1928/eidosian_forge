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
def convert_service_networks(networks):
    if not networks:
        return networks
    if not isinstance(networks, list):
        raise TypeError('networks parameter must be a list.')
    result = []
    for n in networks:
        if isinstance(n, str):
            n = {'Target': n}
        result.append(n)
    return result