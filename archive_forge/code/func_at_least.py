from __future__ import absolute_import, division, print_function
import os
import re
from copy import deepcopy
from datetime import datetime
from ansible.module_utils.urls import urlparse
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import remove_default_spec
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.ipaddress import is_valid_ip, validate_ip_v6_address
from ..module_utils.compare import cmp_str_with_none
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
@property
def at_least(self):
    """Returns the 'at least' value from the monitor string.
        The monitor string for a Require monitor looks like this.
            min 1 of { /Common/gateway_icmp }
        This method parses out the first of the numeric values. This values represents
        the "at_least" value that can be updated in the module.
        Returns:
             int: The at_least value if found. None otherwise.
        """
    if self._values['monitors'] is None:
        return None
    pattern = 'min\\s+(?P<least>\\d+)\\s+of\\s+'
    matches = re.search(pattern, self._values['monitors'])
    if matches is None:
        return None
    return int(matches.group('least'))