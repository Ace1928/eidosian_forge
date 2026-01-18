import time
from libcloud.compute.base import (
from libcloud.utils.iso8601 import parse_date
from libcloud.common.gridscale import GridscaleBaseDriver, GridscaleConnection
from libcloud.compute.providers import Provider
def ex_rename_volume(self, volume, name):
    """
        Modify storage volume name

        :param volume: Storage.
        :type volume: :class:.`StorageVolume`

        :param name: New storage name.
        :type name: ``str``

        :return: ``True`` or ``False``
        :rtype: ``bool``
        """
    result = self._sync_request(data={'name': name}, endpoint='objects/storages/{}'.format(volume.id), method='PATCH')
    return result.status == 204