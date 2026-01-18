from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_if_grp_detail(self, record, current_port_list):
    current = {'node': record['node']['name'], 'uuid': record['uuid'], 'name': record['name'], 'ports': current_port_list}
    if record.get('broadcast_domain'):
        current['broadcast_domain'] = record['broadcast_domain']['name']
        current['ipspace'] = record['broadcast_domain']['ipspace']['name']
    return current