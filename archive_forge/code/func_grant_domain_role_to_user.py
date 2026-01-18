import datetime
from collections import namedtuple
from libcloud.utils.py3 import httplib
from libcloud.common.base import Response, ConnectionUserAndKey, CertificateConnection
from libcloud.compute.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.utils.iso8601 import parse_date
def grant_domain_role_to_user(self, domain, role, user):
    """
        Grant domain role to a user.

        Note: This function appears to be idempotent.

        :param domain: Domain to grant the role to.
        :type domain: :class:`.OpenStackIdentityDomain`

        :param role: Role to grant.
        :type role: :class:`.OpenStackIdentityRole`

        :param user: User to grant the role to.
        :type user: :class:`.OpenStackIdentityUser`

        :return: ``True`` on success.
        :rtype: ``bool``
        """
    path = '/v3/domains/{}/users/{}/roles/{}'.format(domain.id, user.id, role.id)
    response = self.authenticated_request(path, method='PUT')
    return response.status == httplib.NO_CONTENT