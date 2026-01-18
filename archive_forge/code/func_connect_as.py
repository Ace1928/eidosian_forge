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
def connect_as(self, **kwargs):
    """Make a new OpenStackCloud object with new auth context.

        Take the existing settings from the current cloud and construct a new
        OpenStackCloud object with some of the auth settings overridden. This
        is useful for getting an object to perform tasks with as another user,
        or in the context of a different project.

        .. code-block:: python

          conn = openstack.connect(cloud='example')
          # Work normally
          servers = conn.list_servers()
          conn2 = conn.connect_as(username='different-user', password='')
          # Work as different-user
          servers = conn2.list_servers()

        :param kwargs: keyword arguments can contain anything that would
                       normally go in an auth dict. They will override the same
                       settings from the parent cloud as appropriate. Entries
                       that do not want to be overridden can be ommitted.
        """
    if self.config._openstack_config:
        config = self.config._openstack_config
    else:
        config = openstack.config.OpenStackConfig(app_name=self.config._app_name, app_version=self.config._app_version, load_yaml_config=False)
    params = copy.deepcopy(self.config.config)
    params.pop('profile', None)

    def pop_keys(params, auth, name_key, id_key):
        if name_key in auth or id_key in auth:
            params['auth'].pop(name_key, None)
            params['auth'].pop(id_key, None)
    for prefix in ('user', 'project'):
        if prefix == 'user':
            name_key = 'username'
        else:
            name_key = 'project_name'
        id_key = '{prefix}_id'.format(prefix=prefix)
        pop_keys(params, kwargs, name_key, id_key)
        id_key = '{prefix}_domain_id'.format(prefix=prefix)
        name_key = '{prefix}_domain_name'.format(prefix=prefix)
        pop_keys(params, kwargs, name_key, id_key)
    for key, value in kwargs.items():
        params['auth'][key] = value
    cloud_region = config.get_one(**params)
    cloud_region._discovery_cache = self.session._discovery_cache
    cloud_region._name = self.name
    cloud_region.config['profile'] = self.name
    return self.__class__(config=cloud_region)