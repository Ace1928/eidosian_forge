import os.path
import socket
from urllib3.poolmanager import PoolManager, proxy_from_url
from urllib3.response import HTTPResponse
from urllib3.util import parse_url
from urllib3.util import Timeout as TimeoutSauce
from urllib3.util.retry import Retry
from urllib3.exceptions import ClosedPoolError
from urllib3.exceptions import ConnectTimeoutError
from urllib3.exceptions import HTTPError as _HTTPError
from urllib3.exceptions import InvalidHeader as _InvalidHeader
from urllib3.exceptions import MaxRetryError
from urllib3.exceptions import NewConnectionError
from urllib3.exceptions import ProxyError as _ProxyError
from urllib3.exceptions import ProtocolError
from urllib3.exceptions import ReadTimeoutError
from urllib3.exceptions import SSLError as _SSLError
from urllib3.exceptions import ResponseError
from urllib3.exceptions import LocationValueError
from .models import Response
from .compat import urlparse, basestring
from .utils import (DEFAULT_CA_BUNDLE_PATH, extract_zipped_paths,
from .structures import CaseInsensitiveDict
from .cookies import extract_cookies_to_jar
from .exceptions import (ConnectionError, ConnectTimeout, ReadTimeout, SSLError,
from .auth import _basic_auth_str
def cert_verify(self, conn, url, verify, cert):
    """Verify a SSL certificate. This method should not be called from user
        code, and is only exposed for use when subclassing the
        :class:`HTTPAdapter <requests.adapters.HTTPAdapter>`.

        :param conn: The urllib3 connection object associated with the cert.
        :param url: The requested URL.
        :param verify: Either a boolean, in which case it controls whether we verify
            the server's TLS certificate, or a string, in which case it must be a path
            to a CA bundle to use
        :param cert: The SSL certificate to verify.
        """
    if url.lower().startswith('https') and verify:
        cert_loc = None
        if verify is not True:
            cert_loc = verify
        if not cert_loc:
            cert_loc = extract_zipped_paths(DEFAULT_CA_BUNDLE_PATH)
        if not cert_loc or not os.path.exists(cert_loc):
            raise IOError('Could not find a suitable TLS CA certificate bundle, invalid path: {}'.format(cert_loc))
        conn.cert_reqs = 'CERT_REQUIRED'
        if not os.path.isdir(cert_loc):
            conn.ca_certs = cert_loc
        else:
            conn.ca_cert_dir = cert_loc
    else:
        conn.cert_reqs = 'CERT_NONE'
        conn.ca_certs = None
        conn.ca_cert_dir = None
    if cert:
        if not isinstance(cert, basestring):
            conn.cert_file = cert[0]
            conn.key_file = cert[1]
        else:
            conn.cert_file = cert
            conn.key_file = None
        if conn.cert_file and (not os.path.exists(conn.cert_file)):
            raise IOError('Could not find the TLS certificate file, invalid path: {}'.format(conn.cert_file))
        if conn.key_file and (not os.path.exists(conn.key_file)):
            raise IOError('Could not find the TLS key file, invalid path: {}'.format(conn.key_file))