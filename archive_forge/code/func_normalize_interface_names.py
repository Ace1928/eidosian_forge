from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.facts.facts import Facts
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.interfaces import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.utils.utils import (
def normalize_interface_names(self, param):
    if param:
        for _k, val in iteritems(param):
            val['name'] = normalize_interface(val['name'])
    return param