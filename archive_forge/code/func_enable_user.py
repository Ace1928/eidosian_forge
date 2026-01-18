import datetime
from collections import namedtuple
from libcloud.utils.py3 import httplib
from libcloud.common.base import Response, ConnectionUserAndKey, CertificateConnection
from libcloud.compute.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.utils.iso8601 import parse_date
def enable_user(self, user):
    """
        Enable user account.

        Note: This operation appears to be idempotent.

        :param user: User to enable.
        :type user: :class:`.OpenStackIdentityUser`

        :return: User account which has been enabled.
        :rtype: :class:`.OpenStackIdentityUser`
        """
    data = {'enabled': True}
    data = json.dumps({'user': data})
    response = self.authenticated_request('/v3/users/%s' % user.id, data=data, method='PATCH')
    user = self._to_user(data=response.object['user'])
    return user