from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_vserver_audit_config_body_rest(self):
    """
        Vserver audit config body for create and modify with rest API.
        """
    body = {}
    if 'events' in self.parameters:
        body['events'] = self.parameters['events']
    if 'guarantee' in self.parameters:
        body['guarantee'] = self.parameters['guarantee']
    if self.na_helper.safe_get(self.parameters, ['log', 'retention', 'count']):
        body['log.retention.count'] = self.parameters['log']['retention']['count']
    if self.na_helper.safe_get(self.parameters, ['log', 'retention', 'duration']):
        body['log.retention.duration'] = self.parameters['log']['retention']['duration']
    if self.na_helper.safe_get(self.parameters, ['log', 'rotation', 'size']):
        body['log.rotation.size'] = self.parameters['log']['rotation']['size']
    if self.na_helper.safe_get(self.parameters, ['log', 'format']):
        body['log.format'] = self.parameters['log']['format']
    if 'log_path' in self.parameters:
        body['log_path'] = self.parameters['log_path']
    return body