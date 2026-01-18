from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def portset_get_rest(self):
    """
        Get current portset info
        :return: List of current ports if query successful, else return {}
        """
    api = 'protocols/san/portsets'
    query = {'svm.name': self.parameters['vserver'], 'name': self.parameters['resource_name']}
    if self.parameters.get('portset_type'):
        query['protocol'] = self.parameters['portset_type']
    fields = 'interfaces'
    record, error = rest_generic.get_one_record(self.rest_api, api, query, fields)
    if error:
        self.module.fail_json(msg=error)
    current = {}
    if record:
        current['uuid'] = record['uuid']
        if 'interfaces' in record:
            ports = [lif.get('ip', lif.get('fc'))['name'] for lif in record['interfaces']]
            current['ports'] = ports
    if not current and self.parameters['state'] == 'present':
        error_msg = "Error: Portset '%s' does not exist" % self.parameters['resource_name']
        self.module.fail_json(msg=error_msg)
    return current