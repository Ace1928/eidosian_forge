from __future__ import absolute_import, division, print_function
from copy import deepcopy
from re import M, findall
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.argspec.lldp_global.lldp_global import (
def parse_attribs(self, attribs, conf):
    config = {}
    for item in attribs:
        value = utils.parse_conf_arg(conf, item)
        if value:
            config[item] = value.strip("'")
        else:
            config[item] = None
    return utils.remove_empties(config)