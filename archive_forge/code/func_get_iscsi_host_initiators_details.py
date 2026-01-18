from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
import ipaddress
def get_iscsi_host_initiators_details(self, iscsi_host_initiators):
    """ Get the details of existing ISCSI initiators in host"""
    iscsi_initiator_list = []
    for iscsi in iscsi_host_initiators:
        iscsi_initiator_details = self.unity.get_initiator(_id=iscsi.id)
        iscsi_path_list = []
        if iscsi_initiator_details.paths is not None:
            for path in iscsi_initiator_details.paths:
                iscsi_path_list.append({'id': path.id, 'is_logged_in': path.is_logged_in})
        iscsi_initiator_list.append({'id': iscsi_initiator_details.id, 'name': iscsi_initiator_details.initiator_id, 'paths': iscsi_path_list})
    return iscsi_initiator_list