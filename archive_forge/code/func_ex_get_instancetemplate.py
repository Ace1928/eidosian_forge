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
def ex_get_instancetemplate(self, name):
    """
        Return an InstanceTemplate object based on a name and optional zone.

        :param  name: The name of the Instance Template.
        :type   name: ``str``

        :return:  An Instance Template object.
        :rtype:   :class:`GCEInstanceTemplate`
        """
    request = '/global/instanceTemplates/%s' % name
    response = self.connection.request(request, method='GET').object
    return self._to_instancetemplate(response)