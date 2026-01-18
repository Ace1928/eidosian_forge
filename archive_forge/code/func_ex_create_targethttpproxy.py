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
def ex_create_targethttpproxy(self, name, urlmap):
    """
        Create a target HTTP proxy.

        :param  name: Name of target HTTP proxy
        :type   name: ``str``

        :keyword  urlmap: URL map defining the mapping from URl to the
                           backendservice.
        :type     healthchecks: ``str`` or :class:`GCEUrlMap`

        :return:  Target Pool object
        :rtype:   :class:`GCETargetPool`
        """
    targetproxy_data = {'name': name}
    if not hasattr(urlmap, 'name'):
        urlmap = self.ex_get_urlmap(urlmap)
    targetproxy_data['urlMap'] = urlmap.extra['selfLink']
    request = '/global/targetHttpProxies'
    self.connection.async_request(request, method='POST', data=targetproxy_data)
    return self.ex_get_targethttpproxy(name)