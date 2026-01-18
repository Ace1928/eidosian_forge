import datetime
from collections import namedtuple
from libcloud.utils.py3 import httplib
from libcloud.common.base import Response, ConnectionUserAndKey, CertificateConnection
from libcloud.compute.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.utils.iso8601 import parse_date
def grant_project_role_to_user(self, project, role, user):
    """
        Grant project role to a user.

        Note: This function appears to be idempotent.

        :param project: Project to grant the role to.
        :type project: :class:`.OpenStackIdentityDomain`

        :param role: Role to grant.
        :type role: :class:`.OpenStackIdentityRole`

        :param user: User to grant the role to.
        :type user: :class:`.OpenStackIdentityUser`

        :return: ``True`` on success.
        :rtype: ``bool``
        """
    path = '/v3/projects/{}/users/{}/roles/{}'.format(project.id, user.id, role.id)
    response = self.authenticated_request(path, method='PUT')
    return response.status == httplib.NO_CONTENT