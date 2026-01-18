import itertools
from oslo_serialization import jsonutils
import webob
def set_service_catalog_headers(self, auth_ref):
    """Convert service catalog from token object into headers.

        Build headers that represent the catalog - see main
        doc info at start of __init__ file for details of headers to be defined

        :param auth_ref: The token data
        :type auth_ref: keystoneauth.access.AccessInfo
        """
    if not auth_ref.has_service_catalog():
        self.headers.pop(self._SERVICE_CATALOG_HEADER, None)
        return
    catalog = auth_ref.service_catalog.catalog
    if auth_ref.version == 'v3':
        catalog = _normalize_catalog(catalog)
    c = jsonutils.dumps(catalog)
    self.headers[self._SERVICE_CATALOG_HEADER] = c