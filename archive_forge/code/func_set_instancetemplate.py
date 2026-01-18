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
def set_instancetemplate(self, instancetemplate):
    """
        Set the Instance Template for this Instance Group.

        :param  instancetemplate: Instance Template to set.
        :type   instancetemplate: :class:`GCEInstanceTemplate`

        :return:  True if successful
        :rtype:   ``bool``
        """
    return self.driver.ex_instancegroupmanager_set_instancetemplate(manager=self, instancetemplate=instancetemplate)