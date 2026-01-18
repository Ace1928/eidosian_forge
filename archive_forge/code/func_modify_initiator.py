from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.elementsw.plugins.module_utils.netapp_elementsw_module import NaElementSWModule
from ansible_collections.netapp.elementsw.plugins.module_utils.netapp_module import NetAppModule
def modify_initiator(self, initiator, existing_initiator):
    """
        modify initiator
        """
    merged_initiator = existing_initiator.copy()
    del initiator['initiator_id']
    merged_initiator.update(initiator)
    initiator_object = ModifyInitiator(initiator_id=merged_initiator['initiator_id'], alias=merged_initiator['alias'], volume_access_group_id=merged_initiator['volume_access_group_id'], attributes=merged_initiator['attributes'])
    initiator_list = [initiator_object]
    try:
        self.sfe.modify_initiators(initiators=initiator_list)
    except Exception as exception_object:
        self.module.fail_json(msg='Error modifying initiator: %s' % to_native(exception_object), exception=traceback.format_exc())