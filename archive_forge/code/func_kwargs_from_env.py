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
def kwargs_from_env(ssl_version=None, assert_hostname=None, environment=None):
    if not environment:
        environment = os.environ
    host = environment.get('DOCKER_HOST')
    cert_path = environment.get('DOCKER_CERT_PATH') or None
    tls_verify = environment.get('DOCKER_TLS_VERIFY')
    if tls_verify == '':
        tls_verify = False
    else:
        tls_verify = tls_verify is not None
    enable_tls = cert_path or tls_verify
    params = {}
    if host:
        params['base_url'] = host
    if not enable_tls:
        return params
    if not cert_path:
        cert_path = os.path.join(os.path.expanduser('~'), '.docker')
    if not tls_verify and assert_hostname is None:
        assert_hostname = False
    params['tls'] = TLSConfig(client_cert=(os.path.join(cert_path, 'cert.pem'), os.path.join(cert_path, 'key.pem')), ca_cert=os.path.join(cert_path, 'ca.pem'), verify=tls_verify, ssl_version=ssl_version, assert_hostname=assert_hostname)
    return params