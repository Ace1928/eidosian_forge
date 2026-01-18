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
def remove_instances(self, node_list):
    """
        Removes one or more instances from the specified instance group,
        but does not delete those instances.

        Scopes needed - one of the following:
        * https://www.googleapis.com/auth/cloud-platform
        * https://www.googleapis.com/auth/compute

        :param  instancegroup:  The Instance Group where you are
                                removng instances.
        :type   instancegroup: :class:``GCEInstanceGroup``

        :param  node_list: List of nodes to add.
        :type   node_list: ``list`` of :class:`Node` or ``list`` of
                           :class:`GCENode`

        :return:  Return True if successful.
        :rtype: ``bool``
        """
    return self.driver.ex_instancegroup_remove_instances(instancegroup=self, node_list=node_list)