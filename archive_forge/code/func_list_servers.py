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
def list_servers(self, detailed=False, all_projects=False, bare=False, filters=None):
    """List all available servers.

        :param detailed: Whether or not to add detailed additional information.
            Defaults to False.
        :param all_projects: Whether to list servers from all projects or just
            the current auth scoped project.
        :param bare: Whether to skip adding any additional information to the
            server record. Defaults to False, meaning the addresses dict will
            be populated as needed from neutron. Setting to True implies
            detailed = False.
        :param filters: Additional query parameters passed to the API server.
        :returns: A list of compute ``Server`` objects.
        """
    if not filters:
        filters = {}
    return [self._expand_server(server, detailed, bare) for server in self.compute.servers(all_projects=all_projects, **filters)]