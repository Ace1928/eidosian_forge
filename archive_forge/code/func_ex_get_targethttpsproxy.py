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
def ex_get_targethttpsproxy(self, name):
    """
        Returns the specified TargetHttpsProxy resource. Get a list of
        available target HTTPS proxies by making a list() request.

        Scopes needed - one of the following:
        * https://www.googleapis.com/auth/cloud-platform
        * https://www.googleapis.com/auth/compute
        * https://www.googleapis.com/auth/compute.readonly

        :param  name:  Name of the TargetHttpsProxy resource to
                                   return.
        :type   name: ``str``

        :return:  `GCETargetHttpsProxy` object.
        :rtype: :class:`GCETargetHttpsProxy`
        """
    request = '/global/targetHttpsProxies/%s' % name
    response = self.connection.request(request, method='GET').object
    return self._to_targethttpsproxy(response)