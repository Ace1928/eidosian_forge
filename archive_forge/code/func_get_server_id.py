import base64
import functools
import operator
import time
import iso8601
from openstack.cloud import _utils
from openstack.cloud import exc
from openstack.cloud import meta
from openstack.compute.v2._proxy import Proxy
from openstack.compute.v2 import quota_set as _qs
from openstack.compute.v2 import server as _server
from openstack import exceptions
from openstack import utils
def get_server_id(self, name_or_id):
    """Get the ID of a server.

        :param name_or_id:
        :returns: The name of the server if found, else None.
        """
    server = self.get_server(name_or_id, bare=True)
    if server:
        return server['id']
    return None