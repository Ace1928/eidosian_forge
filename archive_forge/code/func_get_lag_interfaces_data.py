from __future__ import absolute_import, division, print_function
from itertools import groupby
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.argspec.lag_interfaces.lag_interfaces import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.lag_interfaces import (
def get_lag_interfaces_data(self, connection):
    return connection.get('show running-config | section ^interface')