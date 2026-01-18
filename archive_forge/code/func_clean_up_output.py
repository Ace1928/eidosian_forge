from __future__ import absolute_import, division, print_function
import copy
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_vserver, zapis_svm
def clean_up_output(self, vserver_details):
    vserver_details['root_volume'] = None
    vserver_details['root_volume_aggregate'] = None
    vserver_details['root_volume_security_style'] = None
    vserver_details['aggr_list'] = [aggr['name'] for aggr in vserver_details['aggregates']]
    vserver_details.pop('aggregates')
    vserver_details['ipspace'] = vserver_details['ipspace']['name']
    vserver_details['snapshot_policy'] = vserver_details['snapshot_policy']['name']
    vserver_details['admin_state'] = vserver_details.pop('state')
    if 'max_volumes' in vserver_details:
        vserver_details['max_volumes'] = str(vserver_details['max_volumes'])
    if vserver_details.get('web') is None and self.parameters.get('web'):
        vserver_details['web'] = {'certificate': {'uuid': vserver_details['certificate']['uuid'] if 'certificate' in vserver_details else None}, 'client_enabled': None, 'ocsp_enabled': None}
    services = {}
    allowed_protocols = None if self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 9, 1) else vserver_details.get('allowed_protocols')
    for protocol in self.allowable_protocols_rest:
        allowed = self.na_helper.safe_get(vserver_details, [protocol, 'allowed'])
        if allowed is None and allowed_protocols is not None:
            allowed = protocol in allowed_protocols
        enabled = self.na_helper.safe_get(vserver_details, [protocol, 'enabled'])
        if allowed is not None or enabled is not None:
            services[protocol] = {}
        if allowed is not None:
            services[protocol]['allowed'] = allowed
        if enabled is not None:
            services[protocol]['enabled'] = enabled
    if services:
        vserver_details['services'] = services
    return vserver_details