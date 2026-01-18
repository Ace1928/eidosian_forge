from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
def get_delete_individual_source_bindings_requests(self, afi, entry):
    """create a request to delete the given source binding entry and address family specified
        by afi"""
    return [{'path': self.binding_uri + '={mac},{ipv}'.format(mac=entry.get('mac_addr'), ipv=afi.get('afi')), 'method': self.delete_method_value}]