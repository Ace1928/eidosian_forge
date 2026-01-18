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
def ex_list(self, list_fn, **kwargs):
    """
        Wrap a list method in a :class:`GCEList` iterator.

        >>> for sublist in driver.ex_list(driver.ex_list_urlmaps).page(1):
        ...   sublist
        ...
        [<GCEUrlMap id="..." name="cli-map">]
        [<GCEUrlMap id="..." name="lc-map">]
        [<GCEUrlMap id="..." name="web-map">]

        :param  list_fn: A bound list method from :class:`GCENodeDriver`.
        :type   list_fn: ``instancemethod``

        :return: An iterator that returns sublists from list_fn.
        :rtype: :class:`GCEList`
        """
    return GCEList(driver=self, list_fn=list_fn, **kwargs)