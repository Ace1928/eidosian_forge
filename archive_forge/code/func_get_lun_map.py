from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
import codecs
from ansible.module_utils._text import to_text, to_bytes
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_lun_map(self):
    """
        Return details about the LUN map

        :return: Details about the lun map
        :rtype: dict
        """
    if self.use_rest:
        return self.get_lun_map_rest()
    lun_info = netapp_utils.zapi.NaElement('lun-map-list-info')
    lun_info.add_new_child('path', self.parameters['path'])
    result = self.server.invoke_successfully(lun_info, True)
    return_value = None
    igroups = result.get_child_by_name('initiator-groups')
    if igroups:
        for igroup_info in igroups.get_children():
            initiator_group_name = igroup_info.get_child_content('initiator-group-name')
            lun_id = igroup_info.get_child_content('lun-id')
            if initiator_group_name == self.parameters['initiator_group_name']:
                return_value = {'lun_id': lun_id}
                break
    return return_value