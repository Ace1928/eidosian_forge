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
def create_lun(self):
    """
        Create LUN with requested name and size
        """
    if self.use_rest:
        return self.create_lun_rest()
    path = '/vol/%s/%s' % (self.parameters['flexvol_name'], self.parameters['name'])
    options = {'path': path, 'size': str(self.parameters['size']), 'space-reservation-enabled': self.na_helper.get_value_for_bool(False, self.parameters['space_reserve']), 'use-exact-size': str(self.parameters['use_exact_size'])}
    if self.parameters.get('space_allocation') is not None:
        options['space-allocation-enabled'] = self.na_helper.get_value_for_bool(False, self.parameters['space_allocation'])
    if self.parameters.get('comment') is not None:
        options['comment'] = self.parameters['comment']
    if self.parameters.get('os_type') is not None:
        options['ostype'] = self.parameters['os_type']
    if self.parameters.get('qos_policy_group') is not None:
        options['qos-policy-group'] = self.parameters['qos_policy_group']
    if self.parameters.get('qos_adaptive_policy_group') is not None:
        options['qos-adaptive-policy-group'] = self.parameters['qos_adaptive_policy_group']
    lun_create = netapp_utils.zapi.NaElement.create_node_with_children('lun-create-by-size', **options)
    try:
        self.server.invoke_successfully(lun_create, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as exc:
        self.module.fail_json(msg='Error provisioning lun %s of size %s: %s' % (self.parameters['name'], self.parameters['size'], to_native(exc)), exception=traceback.format_exc())