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
def ex_get_license(self, project, name):
    """
        Return a License object for specified project and name.

        :param  project: The project to reference when looking up the license.
        :type   project: ``str``

        :param  name: The name of the License
        :type   name: ``str``

        :return:  A License object for the name
        :rtype:   :class:`GCELicense`
        """
    return GCELicense.lazy(name, project, self)