from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.argspec.l3_interfaces.l3_interfaces import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.l3_interfaces import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.utils.utils import (
def get_l3_interfaces_data(self, connection):
    return connection.get('show running-config | section ^interface')