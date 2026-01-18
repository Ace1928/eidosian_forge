from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.aaa.aaa import AaaArgs
def parse_sonic_aaa(self, spec, conf):
    config = deepcopy(spec)
    if conf:
        temp = {}
        if 'authentication-method' in conf and conf['authentication-method']:
            if 'local' in conf['authentication-method']:
                temp['local'] = True
            choices = ['tacacs+', 'ldap', 'radius']
            for i, word in enumerate(conf['authentication-method']):
                if word in choices:
                    temp['group'] = conf['authentication-method'][i]
        if 'failthrough' in conf:
            temp['fail_through'] = conf['failthrough']
        if temp:
            config['authentication']['data'] = temp
    return utils.remove_empties(config)