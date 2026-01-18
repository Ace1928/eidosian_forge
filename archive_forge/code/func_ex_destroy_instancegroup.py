import sys
import time
import datetime
import itertools
from libcloud.pricing import get_pricing
from libcloud.common.base import LazyObject
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.google import (
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
def ex_destroy_instancegroup(self, instancegroup):
    """
        Deletes the specified instance group. The instances in the group
        are not deleted. Note that instance group must not belong to a backend
        service. Read  Deleting an instance group for more information.

        Scopes needed - one of the following:
        * https://www.googleapis.com/auth/cloud-platform
        * https://www.googleapis.com/auth/compute

        :param  instancegroup:  The name of the instance group to delete.
        :type   instancegroup: :class:`GCEInstanceGroup`

        :return:  Return True if successful.
        :rtype: ``bool``
        """
    request = '/zones/{}/instanceGroups/{}'.format(instancegroup.zone.name, instancegroup.name)
    request_data = {}
    self.connection.async_request(request, method='DELETE', data=request_data)
    return True