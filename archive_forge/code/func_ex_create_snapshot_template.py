import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
def ex_create_snapshot_template(self, snapshot, name, ostypeid, displaytext=None):
    """
        Create a template from a snapshot

        :param      snapshot: Instance of ``VolumeSnapshot``
        :type       volume: ``VolumeSnapshot``

        :param  name: the name of the template
        :type   name: ``str``

        :param  name: the os type id
        :type   name: ``str``

        :param  name: the display name of the template
        :type   name: ``str``

        :rtype: :class:`NodeImage`
        """
    if not displaytext:
        displaytext = name
    resp = self._async_request(command='createTemplate', params={'displaytext': displaytext, 'name': name, 'ostypeid': ostypeid, 'snapshotid': snapshot.id})
    img = resp.get('template')
    extra = {'hypervisor': img['hypervisor'], 'format': img['format'], 'os': img['ostypename'], 'displaytext': img['displaytext']}
    return NodeImage(id=img['id'], name=img['name'], driver=self.connection.driver, extra=extra)