from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.argspec.ntp_global.ntp_global import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.ntp_global import (
def get_ntp_data(self, connection):
    return connection.get('show running-config | section ^ntp')