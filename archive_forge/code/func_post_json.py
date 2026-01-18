from __future__ import (absolute_import, division, print_function)
import json
import logging
import struct
from functools import partial
from ansible.module_utils.six import PY3, binary_type, iteritems, string_types, raise_from
from ansible.module_utils.six.moves.urllib.parse import quote
from .. import auth
from .._import_helper import fail_on_missing_imports
from .._import_helper import HTTPError as _HTTPError
from .._import_helper import InvalidSchema as _InvalidSchema
from .._import_helper import Session as _Session
from ..constants import (DEFAULT_NUM_POOLS, DEFAULT_NUM_POOLS_SSH,
from ..errors import (DockerException, InvalidVersion, TLSParameterError, MissingRequirementException,
from ..tls import TLSConfig
from ..transport.npipeconn import NpipeHTTPAdapter
from ..transport.npipesocket import PYWIN32_IMPORT_ERROR
from ..transport.unixconn import UnixHTTPAdapter
from ..transport.sshconn import SSHHTTPAdapter, PARAMIKO_IMPORT_ERROR
from ..transport.ssladapter import SSLHTTPAdapter
from ..utils import config, utils, json_stream
from ..utils.decorators import check_resource, update_headers
from ..utils.proxy import ProxyConfig
from ..utils.socket import consume_socket_output, demux_adaptor, frames_iter
from .daemon import DaemonApiMixin
def post_json(self, pathfmt, *args, **kwargs):
    data = kwargs.pop('data', None)
    self._raise_for_status(self._post_json(self._url(pathfmt, *args, versioned_api=True), data, **kwargs))