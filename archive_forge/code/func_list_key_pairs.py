import copy
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.common.types import ProviderError
from libcloud.compute.base import (
from libcloud.compute.types import NodeState, VolumeSnapshotState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
def list_key_pairs(self):
    """
        List all the available SSH keys.

        :return: Available SSH keys.
        :rtype: ``list`` of :class:`KeyPair`
        """
    response = self.connection.request('/users/%s' % self._get_user_id(), region='account')
    keys = response.object['user']['ssh_public_keys']
    return [KeyPair(name=' '.join(key['key'].split(' ')[2:]), public_key=' '.join(key['key'].split(' ')[:2]), fingerprint=key['fingerprint'], driver=self) for key in keys]