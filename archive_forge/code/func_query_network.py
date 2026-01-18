from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.api import (
def query_network(self, uuid=None):
    if self._network is not None:
        return self._network
    net_param = self._module.params['network']
    net_uuid = uuid or net_param['uuid']
    if net_uuid is not None:
        network = self._get('networks/%s' % net_uuid)
        if not network:
            self._module.fail_json(msg="Network with 'uuid' not found: %s" % net_uuid)
    elif net_param['name'] is not None:
        networks_found = []
        networks = self._get('networks')
        for network in networks or []:
            if net_param['zone'] is not None and network['zone']['slug'] != net_param['zone']:
                continue
            if network.get('name') == net_param['name']:
                networks_found.append(network)
        if not networks_found:
            msg = "Network with 'name' not found: %s" % net_param['name']
            self._module.fail_json(msg=msg)
        elif len(networks_found) == 1:
            network = networks_found[0]
        else:
            msg = "Multiple networks with 'name' not found: %s.Add the 'zone' to distinguish or use 'uuid' argument to specify the network." % net_param['name']
            self._module.fail_json(msg=msg)
    else:
        self._module.fail_json(msg='Either Network UUID or name is required.')
    self._network = dict()
    for k, v in network.items():
        if k in ['name', 'uuid', 'href', 'zone']:
            self._network[k] = v
    return self._network