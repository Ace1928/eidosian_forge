import time
from libcloud.compute.base import (
from libcloud.utils.iso8601 import parse_date
from libcloud.common.gridscale import GridscaleBaseDriver, GridscaleConnection
from libcloud.compute.providers import Provider
def ex_storage_rollback(self, volume, snapshot, rollback):
    """
        initiate a rollback on your storage

        :param volume: storage uuid
        :type volume: ``string``

        :param snapshot: snapshot uuid
        :type snapshot: ``string``

        :param rollback: variable
        :type rollback: ``bool``

        :return: RequestID
        :rtype: ``str``
        """
    result = self._sync_request(data={'rollback': rollback}, endpoint='objects/storages/{}/snapshots/{}/rollback'.format(volume.id, snapshot.id), method='PATCH')
    return result