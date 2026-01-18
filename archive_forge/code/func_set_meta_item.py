import base64
import collections
from urllib import parse
from novaclient import api_versions
from novaclient import base
from novaclient import crypto
from novaclient import exceptions
from novaclient.i18n import _
def set_meta_item(self, server, key, value):
    """
        Updates an item of server metadata
        :param server: The :class:`Server` to add metadata to
        :param key: metadata key to update
        :param value: string value
        """
    body = {'meta': {key: value}}
    return self._update('/servers/%s/metadata/%s' % (base.getid(server), key), body)