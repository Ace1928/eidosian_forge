import logging
import re
import sys
import warnings
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.exceptions import UnsupportedAlgorithm
from requests.auth import AuthBase
from requests.models import Response
from requests.compat import urlparse, StringIO
from requests.structures import CaseInsensitiveDict
from requests.cookies import cookiejar_from_dict
from requests.packages.urllib3 import HTTPResponse
from .exceptions import MutualAuthenticationError, KerberosExchangeError
def authenticate_server(self, response):
    """
        Uses GSSAPI to authenticate the server.

        Returns True on success, False on failure.
        """
    log.debug('authenticate_server(): Authenticate header: {0}'.format(_negotiate_value(response)))
    host = urlparse(response.url).hostname
    try:
        if self.cbt_struct:
            result = kerberos.authGSSClientStep(self.context[host], _negotiate_value(response), channel_bindings=self.cbt_struct)
        else:
            result = kerberos.authGSSClientStep(self.context[host], _negotiate_value(response))
    except kerberos.GSSError:
        log.exception('authenticate_server(): authGSSClientStep() failed:')
        return False
    if result < 1:
        log.error('authenticate_server(): authGSSClientStep() failed: {0}'.format(result))
        return False
    log.debug('authenticate_server(): returning {0}'.format(response))
    return True