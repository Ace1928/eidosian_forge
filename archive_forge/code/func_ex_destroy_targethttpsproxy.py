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
def ex_destroy_targethttpsproxy(self, targethttpsproxy):
    """
        Deletes the specified TargetHttpsProxy resource.

        Scopes needed - one of the following:
        * https://www.googleapis.com/auth/cloud-platform
        * https://www.googleapis.com/auth/compute

        :param  targethttpsproxy:  Name of the TargetHttpsProxy resource to
                                   delete.
        :type   targethttpsproxy: ``str``

        :return  targetHttpsProxy:  Return True if successful.
        :rtype   targetHttpsProxy: ````bool````
        """
    request = '/global/targetHttpsProxies/%s' % targethttpsproxy.name
    request_data = {}
    self.connection.async_request(request, method='DELETE', data=request_data)
    return True