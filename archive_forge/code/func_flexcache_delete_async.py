from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_volume
def flexcache_delete_async(self):
    """
        Delete FlexCache relationship at destination cluster
        """
    options = {'volume': self.parameters['name']}
    flexcache_delete = netapp_utils.zapi.NaElement.create_node_with_children('flexcache-destroy-async', **options)
    try:
        result = self.server.invoke_successfully(flexcache_delete, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error deleting FlexCache: %s' % to_native(error), exception=traceback.format_exc())
    results = {}
    for key in ('result-status', 'result-jobid'):
        if result.get_child_by_name(key):
            results[key] = result[key]
    return results