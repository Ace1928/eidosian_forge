from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.facts.facts import Facts
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.route_maps import (
def convert_to_dict(inner_match, key):
    temp = dict()
    for each in inner_match:
        temp.update({key + '_' + str(each): each})
    return dict(sorted(temp.items(), key=lambda x: x[1]))