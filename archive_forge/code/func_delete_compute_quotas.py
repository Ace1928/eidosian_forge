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
def delete_compute_quotas(self, name_or_id):
    """Delete quota for a project

        :param name_or_id: project name or id

        :raises: :class:`~openstack.exceptions.SDKException` if it's not a
            valid project or the nova client call failed
        """
    proj = self.identity.find_project(name_or_id, ignore_missing=False)
    self.compute.revert_quota_set(proj)