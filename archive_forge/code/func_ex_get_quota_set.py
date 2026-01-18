import base64
import warnings
from libcloud.pricing import get_size_price
from libcloud.utils.py3 import ET, b, next, httplib, parse_qs, urlparse
from libcloud.utils.xml import findall
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.iso8601 import parse_date
from libcloud.common.openstack import (
from libcloud.utils.networking import is_public_subnet
from libcloud.common.exceptions import BaseHTTPError
def ex_get_quota_set(self, tenant_id, user_id=None):
    """
        Get the quota for a project or a project and a user.

        :param      tenant_id: The UUID of the tenant in a multi-tenancy cloud
        :type       tenant_id: ``str``

        :param      user_id: ID of user to list the quotas for.
        :type       user_id: ``str``

        :rtype: :class:`OpenStack_2_QuotaSet`
        """
    url = '/os-quota-sets/%s/detail' % tenant_id
    if user_id:
        url += '?user_id=%s' % user_id
    return self._to_quota_set(self.connection.request(url).object['quota_set'])