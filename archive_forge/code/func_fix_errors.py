from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, netapp_ipaddress
def fix_errors(self, options, errors):
    """ignore role and firewall_policy if a service_policy can be safely derived"""
    block_p, file_p, fcp = self.derive_block_file_type(self.parameters.get('protocols'))
    if 'role' in errors:
        fixed = False
        if errors['role'] == 'data' and errors.get('firewall_policy', 'data') == 'data':
            if fcp:
                fixed = True
            elif file_p and self.parameters.get('service_policy', 'default-data-files') == 'default-data-files':
                options['service_policy'] = 'default-data-files'
                fixed = True
            elif block_p and self.parameters.get('service_policy', 'default-data-blocks') == 'default-data-blocks':
                options['service_policy'] = 'default-data-blocks'
                fixed = True
        if errors['role'] == 'data' and errors.get('firewall_policy') == 'mgmt':
            options['service_policy'] = 'default-management'
            fixed = True
        if errors['role'] == 'intercluster' and errors.get('firewall_policy') in [None, 'intercluster']:
            options['service_policy'] = 'default-intercluster'
            fixed = True
        if errors['role'] == 'cluster' and errors.get('firewall_policy') in [None, 'mgmt']:
            options['service_policy'] = 'default-cluster'
            fixed = True
        if errors['role'] == 'data' and fcp and (errors.get('firewall_policy') is None):
            fixed = True
        if fixed:
            errors.pop('role')
            errors.pop('firewall_policy', None)