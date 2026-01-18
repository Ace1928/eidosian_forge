from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, netapp_ipaddress
def set_protocol_option(self, required_keys):
    """ set protocols for create """
    if self.parameters.get('protocols') is None:
        return None
    data_protocols_obj = netapp_utils.zapi.NaElement('data-protocols')
    for protocol in self.parameters.get('protocols'):
        if protocol.lower() in ['fc-nvme', 'fcp']:
            if 'address' in required_keys:
                required_keys.remove('address')
            if 'home_port' in required_keys:
                required_keys.remove('home_port')
            if 'netmask' in required_keys:
                required_keys.remove('netmask')
            not_required_params = set(['address', 'netmask', 'firewall_policy'])
            if not not_required_params.isdisjoint(set(self.parameters.keys())):
                self.module.fail_json(msg='Error: Following parameters for creating interface are not supported for data-protocol fc-nvme: %s' % ', '.join(not_required_params))
        data_protocols_obj.add_new_child('data-protocol', protocol)
    return data_protocols_obj