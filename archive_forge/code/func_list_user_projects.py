import datetime
from collections import namedtuple
from libcloud.utils.py3 import httplib
from libcloud.common.base import Response, ConnectionUserAndKey, CertificateConnection
from libcloud.compute.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.utils.iso8601 import parse_date
def list_user_projects(self, user):
    """
        Retrieve all the projects user belongs to.

        :rtype: ``list`` of :class:`.OpenStackIdentityProject`
        """
    path = '/v3/users/%s/projects' % user.id
    response = self.authenticated_request(path, method='GET')
    result = self._to_projects(data=response.object['projects'])
    return result