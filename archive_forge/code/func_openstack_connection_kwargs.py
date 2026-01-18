from libcloud.utils.py3 import ET, httplib
from libcloud.common.base import Response, ConnectionUserAndKey
from libcloud.common.types import ProviderError
from libcloud.compute.types import LibcloudError, MalformedResponseError, KeyPairDoesNotExistError
from libcloud.common.exceptions import BaseHTTPError
from libcloud.common.openstack_identity import (
def openstack_connection_kwargs(self):
    """
        Returns certain ``ex_*`` parameters for this connection.

        :rtype: ``dict``
        """
    rv = {}
    if self._ex_force_base_url:
        rv['ex_force_base_url'] = self._ex_force_base_url
    if self._ex_force_auth_token:
        rv['ex_force_auth_token'] = self._ex_force_auth_token
    if self._ex_force_auth_url:
        rv['ex_force_auth_url'] = self._ex_force_auth_url
    if self._ex_force_auth_version:
        rv['ex_force_auth_version'] = self._ex_force_auth_version
    if self._ex_token_scope:
        rv['ex_token_scope'] = self._ex_token_scope
    if self._ex_domain_name:
        rv['ex_domain_name'] = self._ex_domain_name
    if self._ex_tenant_name:
        rv['ex_tenant_name'] = self._ex_tenant_name
    if self._ex_tenant_domain_id:
        rv['ex_tenant_domain_id'] = self._ex_tenant_domain_id
    if self._ex_force_service_type:
        rv['ex_force_service_type'] = self._ex_force_service_type
    if self._ex_force_service_name:
        rv['ex_force_service_name'] = self._ex_force_service_name
    if self._ex_force_service_region:
        rv['ex_force_service_region'] = self._ex_force_service_region
    if self._ex_auth_cache is not None:
        rv['ex_auth_cache'] = self._ex_auth_cache
    if self._ex_force_microversion:
        rv['ex_force_microversion'] = self._ex_force_microversion
    return rv