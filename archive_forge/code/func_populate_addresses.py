from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.routeros.plugins.module_utils.api import (
def populate_addresses(self, data, family):
    for value in data:
        key = value['interface']
        if family not in self.facts['interfaces'][key]:
            self.facts['interfaces'][key][family] = []
        addr, subnet = value['address'].split('/')
        subnet = subnet.strip()
        try:
            subnet = int(subnet)
        except Exception:
            pass
        ip = dict(address=addr.strip(), subnet=subnet)
        self.add_ip_address(addr.strip(), family)
        self.facts['interfaces'][key][family].append(ip)