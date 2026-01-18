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
def parse_env_file(env_file):
    """
    Reads a line-separated environment file.
    The format of each line should be "key=value".
    """
    environment = {}
    with open(env_file) as f:
        for line in f:
            if line[0] == '#':
                continue
            line = line.strip()
            if not line:
                continue
            parse_line = line.split('=', 1)
            if len(parse_line) == 2:
                k, v = parse_line
                environment[k] = v
            else:
                raise errors.DockerException('Invalid line in environment file {}:\n{}'.format(env_file, line))
    return environment