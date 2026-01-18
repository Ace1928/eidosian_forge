from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_broadcast_domain_ports_rest(self):
    """
        Return details about the broadcast domain ports.
        :return: Details about the broadcast domain ports. [] if not found.
        :rtype: list
        """
    api = 'network/ethernet/broadcast-domains'
    query = {'name': self.parameters['resource_name'], 'ipspace.name': self.parameters['ipspace']}
    fields = 'ports'
    record, error = rest_generic.get_one_record(self.rest_api, api, query, fields)
    if error:
        self.module.fail_json(msg=error)
    ports = []
    if record and 'ports' in record:
        ports = ['%s:%s' % (port['node']['name'], port['name']) for port in record['ports']]
    return ports