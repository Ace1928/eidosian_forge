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
def get_compute_limits(self, name_or_id=None):
    """Get absolute compute limits for a project

        :param name_or_id: (optional) project name or ID to get limits for
            if different from the current project

        :returns: A compute
            :class:`~openstack.compute.v2.limits.Limits.AbsoluteLimits` object.
        :raises: :class:`~openstack.exceptions.SDKException` if it's not a
            valid project
        """
    params = {}
    project_id = None
    if name_or_id:
        proj = self.get_project(name_or_id)
        if not proj:
            raise exceptions.SDKException('project does not exist')
        project_id = proj.id
        params['tenant_id'] = project_id
    return self.compute.get_limits(**params).absolute