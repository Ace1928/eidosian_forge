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
def get_server_meta(self, server):
    """Get the metadata for a server.

        :param server:
        :returns: The metadata for the server if found, else None.
        """
    server_vars = meta.get_hostvars_from_server(self, server)
    groups = meta.get_groups_from_server(self, server, server_vars)
    return dict(server_vars=server_vars, groups=groups)