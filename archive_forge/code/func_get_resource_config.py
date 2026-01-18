from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.junipernetworks.junos.plugins.module_utils.network.junos.junos import (
def get_resource_config(connection, config_filter=None, attrib=None):
    if attrib is None:
        attrib = {'inherit': 'inherit'}
    get_ele = new_ele('get-configuration', attrib)
    if config_filter:
        get_ele.append(to_ele(config_filter))
    return connection.execute_rpc(tostring(get_ele))