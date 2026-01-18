import datetime
from collections import namedtuple
from libcloud.utils.py3 import httplib
from libcloud.common.base import Response, ConnectionUserAndKey, CertificateConnection
from libcloud.compute.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.utils.iso8601 import parse_date
def list_user_domain_roles(self, domain, user):
    """
        Retrieve all the roles for a particular user on a domain.

        :rtype: ``list`` of :class:`.OpenStackIdentityRole`
        """
    path = '/v3/domains/{}/users/{}/roles'.format(domain.id, user.id)
    response = self.authenticated_request(path, method='GET')
    result = self._to_roles(data=response.object['roles'])
    return result