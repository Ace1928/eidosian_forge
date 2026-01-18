from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_volume
def flexcache_create_async(self):
    """
        Create a FlexCache relationship
        """
    options = {'origin-volume': self.parameters['origin_volume'], 'origin-vserver': self.parameters['origin_vserver'], 'volume': self.parameters['name']}
    self.add_parameter_to_dict(options, 'junction_path', 'junction-path')
    self.add_parameter_to_dict(options, 'auto_provision_as', 'auto-provision-as')
    self.add_parameter_to_dict(options, 'size', 'size', tostr=True)
    if self.parameters.get('aggr_list') and self.parameters.get('aggr_list_multiplier'):
        self.add_parameter_to_dict(options, 'aggr_list_multiplier', 'aggr-list-multiplier', tostr=True)
    flexcache_create = netapp_utils.zapi.NaElement.create_node_with_children('flexcache-create-async', **options)
    if self.parameters.get('aggr_list'):
        aggregates = netapp_utils.zapi.NaElement('aggr-list')
        for aggregate in self.parameters['aggr_list']:
            aggregates.add_new_child('aggr-name', aggregate)
        flexcache_create.add_child_elem(aggregates)
    try:
        result = self.server.invoke_successfully(flexcache_create, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error creating FlexCache: %s' % to_native(error), exception=traceback.format_exc())
    results = {}
    for key in ('result-status', 'result-jobid'):
        if result.get_child_by_name(key):
            results[key] = result[key]
    return results