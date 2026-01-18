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
def connect_as_project(self, project):
    """Make a new OpenStackCloud object with a new project.

        Take the existing settings from the current cloud and construct a new
        OpenStackCloud object with the project settings overridden. This
        is useful for getting an object to perform tasks with as another user,
        or in the context of a different project.

        .. code-block:: python

          cloud = openstack.connect(cloud='example')
          # Work normally
          servers = cloud.list_servers()
          cloud2 = cloud.connect_as_project('different-project')
          # Work in different-project
          servers = cloud2.list_servers()

        :param project: Either a project name or a project dict as returned by
                        `list_projects`.
        """
    auth = {}
    if isinstance(project, dict):
        auth['project_id'] = project.get('id')
        auth['project_name'] = project.get('name')
        if project.get('domain_id'):
            auth['project_domain_id'] = project['domain_id']
    else:
        auth['project_name'] = project
    return self.connect_as(**auth)