from __future__ import absolute_import, division, print_function
import copy
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.rest_application import RestApplication
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_volume
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def check_for_errors(self, lun_cd_action, current, modify):
    errors = []
    if lun_cd_action == 'create':
        if self.parameters.get('flexvol_name') is None:
            errors.append('The flexvol_name parameter is required for creating a LUN.')
        if self.use_rest and self.parameters.get('os_type') is None:
            errors.append('The os_type parameter is required for creating a LUN with REST.')
        if self.parameters.get('size') is None:
            self.module.fail_json(msg='size is a required parameter for create.')
    elif modify and 'os_type' in modify:
        self.module.fail_json(msg='os_type cannot be modified: current: %s, desired: %s' % (current['os_type'], modify['os_type']))
    if errors:
        self.module.fail_json(msg=' '.join(errors))