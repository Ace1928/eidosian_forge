import copy
import functools
import queue
import warnings
import dogpile.cache
import keystoneauth1.exceptions
import keystoneauth1.session
import requests.models
import requestsexceptions
from openstack import _log
from openstack.cloud import _object_store
from openstack.cloud import _utils
from openstack.cloud import meta
import openstack.config
from openstack.config import cloud_region as cloud_region_mod
from openstack import exceptions
from openstack import proxy
from openstack import utils
def global_request(self, global_request_id):
    """Make a new Connection object with a global request id set.

        Take the existing settings from the current Connection and construct a
        new Connection object with the global_request_id overridden.

        .. code-block:: python

          from oslo_context import context
          cloud = openstack.connect(cloud='example')
          # Work normally
          servers = cloud.list_servers()
          cloud2 = cloud.global_request(context.generate_request_id())
          # cloud2 sends all requests with global_request_id set
          servers = cloud2.list_servers()

        Additionally, this can be used as a context manager:

        .. code-block:: python

          from oslo_context import context
          c = openstack.connect(cloud='example')
          # Work normally
          servers = c.list_servers()
          with c.global_request(context.generate_request_id()) as c2:
              # c2 sends all requests with global_request_id set
              servers = c2.list_servers()

        :param global_request_id: The `global_request_id` to send.
        """
    params = copy.deepcopy(self.config.config)
    cloud_region = cloud_region_mod.from_session(session=self.session, app_name=self.config._app_name, app_version=self.config._app_version, discovery_cache=self.session._discovery_cache, **params)
    cloud_region._name = self.name
    cloud_region.config['profile'] = self.name
    new_conn = self.__class__(config=cloud_region)
    new_conn.set_global_request_id(global_request_id)
    return new_conn