import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
def ex_register_iso(self, name, url, location=None, **kwargs):
    """
        Registers an existing ISO by URL.

        :param      name: Name which should be used
        :type       name: ``str``

        :param      url: Url should be used
        :type       url: ``str``

        :param      location: Location which should be used
        :type       location: :class:`NodeLocation`

        :rtype: ``str``
        """
    if location is None:
        location = self.list_locations()[0]
    params = {'name': name, 'displaytext': name, 'url': url, 'zoneid': location.id}
    params['bootable'] = kwargs.pop('bootable', False)
    if params['bootable']:
        os_type_id = kwargs.pop('ostypeid', None)
        if not os_type_id:
            raise LibcloudError('If bootable=True, ostypeid is required!')
        params['ostypeid'] = os_type_id
    return self._sync_request(command='registerIso', params=params)