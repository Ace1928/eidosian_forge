from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def does_snapshot_exist(self, volume):
    """
        This is duplicated from na_ontap_snapshot
        Checks to see if a snapshot exists or not
        :return: Return True if a snapshot exists, false if it dosen't
        """
    snapshot_obj = netapp_utils.zapi.NaElement('snapshot-get-iter')
    desired_attr = netapp_utils.zapi.NaElement('desired-attributes')
    snapshot_info = netapp_utils.zapi.NaElement('snapshot-info')
    comment = netapp_utils.zapi.NaElement('comment')
    snapshot_info.add_child_elem(comment)
    desired_attr.add_child_elem(snapshot_info)
    snapshot_obj.add_child_elem(desired_attr)
    query = netapp_utils.zapi.NaElement('query')
    snapshot_info_obj = netapp_utils.zapi.NaElement('snapshot-info')
    snapshot_info_obj.add_new_child('name', self.parameters['snapshot'])
    snapshot_info_obj.add_new_child('volume', volume)
    snapshot_info_obj.add_new_child('vserver', self.parameters['vserver'])
    query.add_child_elem(snapshot_info_obj)
    snapshot_obj.add_child_elem(query)
    result = self.server.invoke_successfully(snapshot_obj, True)
    return_value = None
    if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) == 1:
        attributes_list = result.get_child_by_name('attributes-list')
        snap_info = attributes_list.get_child_by_name('snapshot-info')
        return_value = {'comment': snap_info.get_child_content('comment')}
    return return_value