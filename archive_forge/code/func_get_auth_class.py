from libcloud.utils.py3 import ET, httplib
from libcloud.common.base import Response, ConnectionUserAndKey
from libcloud.common.types import ProviderError
from libcloud.compute.types import LibcloudError, MalformedResponseError, KeyPairDoesNotExistError
from libcloud.common.exceptions import BaseHTTPError
from libcloud.common.openstack_identity import (
def get_auth_class(self):
    """
        Retrieve identity / authentication class instance.

        :rtype: :class:`OpenStackIdentityConnection`
        """
    if not self._osa:
        auth_url = self._get_auth_url()
        cls = get_class_for_auth_version(auth_version=self._auth_version)
        self._osa = cls(auth_url=auth_url, user_id=self.user_id, key=self.key, tenant_name=self._ex_tenant_name, tenant_domain_id=self._ex_tenant_domain_id, domain_name=self._ex_domain_name, token_scope=self._ex_token_scope, timeout=self.timeout, proxy_url=self.proxy_url, parent_conn=self, auth_cache=self._ex_auth_cache)
    return self._osa