import itertools
from oslo_serialization import jsonutils
import webob
def set_user_headers(self, auth_ref):
    """Convert token object into headers.

        Build headers that represent authenticated user - see main
        doc info at start of __init__ file for details of headers to be defined
        """
    self._set_auth_headers(auth_ref, self._USER_HEADER_PREFIX)
    self.headers[self._ADMIN_PROJECT_HEADER] = _is_admin_project(auth_ref)
    for k, v in self._DEPRECATED_HEADER_MAP.items():
        self.headers[k] = self.headers[v]