from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, netapp_ipaddress
def build_rest_payloads(self, cd_action, modify, current):
    body, migrate_body = (None, None)
    uuid = current.get('uuid') if current else None
    if self.use_rest:
        if cd_action == 'create':
            body, migrate_body = self.build_rest_body()
        elif modify:
            if modify.get('home_node') and (not modify.get('home_port')) and (self.parameters['interface_type'] == 'fc'):
                modify['home_port'] = current['home_port']
            if modify.get('current_node') and (not modify.get('current_port')) and (self.parameters['interface_type'] == 'fc'):
                modify['current_port'] = current['current_port']
            body, migrate_body = self.build_rest_body(modify)
        if (modify or cd_action == 'delete') and uuid is None:
            self.module.fail_json(msg='Error, expecting uuid in existing record')
        desired_home_port = self.na_helper.safe_get(body, ['location', 'home_port'])
        desired_current_port = self.na_helper.safe_get(migrate_body, ['location', 'port'])
        if self.parameters.get('interface_type') == 'fc' and desired_home_port and desired_current_port and (desired_home_port == desired_current_port):
            migrate_body = None
    return (uuid, body, migrate_body)