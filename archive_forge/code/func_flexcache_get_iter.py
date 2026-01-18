from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_volume
def flexcache_get_iter(self):
    """
        Compose NaElement object to query current FlexCache relation
        """
    options = {'volume': self.parameters['name']}
    self.add_parameter_to_dict(options, 'origin_volume', 'origin-volume')
    self.add_parameter_to_dict(options, 'origin_vserver', 'origin-vserver')
    self.add_parameter_to_dict(options, 'origin_cluster', 'origin-cluster')
    flexcache_info = netapp_utils.zapi.NaElement.create_node_with_children('flexcache-info', **options)
    query = netapp_utils.zapi.NaElement('query')
    query.add_child_elem(flexcache_info)
    flexcache_get_iter = netapp_utils.zapi.NaElement('flexcache-get-iter')
    flexcache_get_iter.add_child_elem(query)
    return flexcache_get_iter