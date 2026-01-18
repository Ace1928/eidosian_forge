from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def get_volume_clone(self):
    if self.use_rest:
        return self.get_volume_clone_rest()
    clone_obj = netapp_utils.zapi.NaElement('volume-clone-get')
    clone_obj.add_new_child('volume', self.parameters['name'])
    try:
        results = self.vserver.invoke_successfully(clone_obj, True)
    except netapp_utils.zapi.NaApiError as error:
        if to_native(error.code) == '15661':
            return None
        self.module.fail_json(msg='Error fetching volume clone information %s: %s' % (self.parameters['name'], to_native(error)))
    current = None
    if results.get_child_by_name('attributes'):
        attributes = results.get_child_by_name('attributes')
        info = attributes.get_child_by_name('volume-clone-info')
        current = {'split': bool(info.get_child_by_name('block-percentage-complete') or info.get_child_by_name('blocks-scanned') or info.get_child_by_name('blocks-updated'))}
    return current