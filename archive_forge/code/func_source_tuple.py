from __future__ import absolute_import, division, print_function
import os
import re
import traceback
from collections import namedtuple
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.constants import (
from ..module_utils.compare import cmp_simple_list
from ..module_utils.icontrol import (
from ..module_utils.ipaddress import (
from ..module_utils.teem import send_teem
@property
def source_tuple(self):
    Source = namedtuple('Source', ['ip', 'route_domain', 'cidr'])
    if self._values['source'] is None:
        result = Source(ip=None, route_domain=None, cidr=None)
        return result
    pattern = '(?P<ip>[^%]+)%(?P<route_domain>[0-9]+)/(?P<cidr>[0-9]+)'
    matches = re.search(pattern, self._values['source'])
    if matches:
        result = Source(ip=matches.group('ip'), route_domain=matches.group('route_domain'), cidr=matches.group('cidr'))
        return result
    pattern = '(?P<ip>[^%]+)/(?P<cidr>[0-9]+)'
    matches = re.search(pattern, self._values['source'])
    if matches:
        result = Source(ip=matches.group('ip'), route_domain=None, cidr=matches.group('cidr'))
        return result
    result = Source(ip=None, route_domain=None, cidr=None)
    return result