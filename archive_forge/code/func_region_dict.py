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
@property
def region_dict(self):
    if self._region_dict is None:
        regions = self.ex_list_regions()
        self._region_dict = {region.name: region for region in regions}
    return self._region_dict