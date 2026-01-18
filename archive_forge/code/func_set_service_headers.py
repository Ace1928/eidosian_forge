import itertools
from oslo_serialization import jsonutils
import webob
def set_service_headers(self, auth_ref):
    """Convert token object into service headers.

        Build headers that represent authenticated user - see main
        doc info at start of __init__ file for details of headers to be defined
        """
    self._set_auth_headers(auth_ref, self._SERVICE_HEADER_PREFIX)