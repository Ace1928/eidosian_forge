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
def recreate_instances(self):
    """
        Recreate instances in a Managed Instance Group.

        :return:  ``list`` of ``dict`` containing instance URI and
                  currentAction. See
                  ex_instancegroupmanager_list_managed_instances for
                  more details.
        :rtype: ``list``
        """
    return self.driver.ex_instancegroupmanager_recreate_instances(manager=self)