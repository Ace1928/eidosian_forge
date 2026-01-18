from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def fc_initiators_result_list(entity):
    """ Get the WWN and id associated with the Unity FC initiators """
    result = []
    if entity:
        LOG.info(SUCCESSFULL_LISTED_MSG)
        for item in entity:
            result.append({'WWN': item.initiator_id, 'id': item.id})
        return result
    else:
        return None