import copy
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.common.types import ProviderError
from libcloud.compute.base import (
from libcloud.compute.types import NodeState, VolumeSnapshotState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
def import_key_pair_from_string(self, name, key_material):
    """
        Import a new public key from string.

        :param name: Key pair name.
        :type name: ``str``

        :param key_material: Public key material.
        :type key_material: ``str``

        :return: Imported key pair object.
        :rtype: :class:`KeyPair`
        """
    new_key = KeyPair(name=name, public_key=' '.join(key_material.split(' ')[:2]), fingerprint=None, driver=self)
    keys = [key for key in self.list_key_pairs() if not key.name == name]
    keys.append(new_key)
    return self._save_keys(keys)