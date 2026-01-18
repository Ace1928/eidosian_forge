from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
def get_nsp(self, name=None):
    if not self.nsps:
        args = {'physicalnetworkid': self.get_physical_network(key='id')}
        res = self.query_api('listNetworkServiceProviders', **args)
        self.nsps = res['networkserviceprovider']
    names = []
    for nsp in self.nsps:
        names.append(nsp['name'])
        if nsp['name'].lower() == name.lower():
            return nsp
    self.module.fail_json(msg="Failed: '{0}' not in network service providers list '[{1}]'".format(name, names))