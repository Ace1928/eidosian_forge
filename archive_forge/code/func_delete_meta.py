import base64
import collections
from urllib import parse
from novaclient import api_versions
from novaclient import base
from novaclient import crypto
from novaclient import exceptions
from novaclient.i18n import _
def delete_meta(self, server, keys):
    """
        Delete metadata from a server

        :param server: The :class:`Server` to add metadata to
        :param keys: A list of metadata keys to delete from the server
        :returns: An instance of novaclient.base.TupleWithMeta
        """
    result = base.TupleWithMeta((), None)
    for k in keys:
        ret = self._delete('/servers/%s/metadata/%s' % (base.getid(server), k))
        result.append_request_ids(ret.request_ids)
    return result