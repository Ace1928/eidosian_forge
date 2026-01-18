from __future__ import absolute_import, division, print_function
import logging
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def is_cg_modified(self, cg_details):
    """Check if the desired consistency group state is different from
            existing consistency group.
            :param cg_details: The dict containing consistency group details
            :return: Boolean value to indicate if modification is needed
        """
    modified = False
    if self.module.params['tiering_policy'] and cg_details['luns'] is None and (self.module.params['volumes'] is None):
        self.module.fail_json(msg='The system cannot assign a tiering policy to an empty consistency group.')
    if self.module.params['hosts'] and cg_details['luns'] is None and (self.module.params['volumes'] is None):
        self.module.fail_json(msg='The system cannot assign hosts to an empty consistency group.')
    if (cg_details['description'] is not None and self.module.params['description'] is not None and (cg_details['description'] != self.module.params['description']) or (cg_details['description'] is None and self.module.params['description'] is not None)) or (cg_details['snap_schedule'] is not None and self.module.params['snap_schedule'] is not None and (cg_details['snap_schedule']['UnitySnapSchedule']['name'] != self.module.params['snap_schedule']) or (cg_details['snap_schedule'] is None and self.module.params['snap_schedule'])):
        modified = True
    if cg_details['relocation_policy']:
        tier_policy = cg_details['relocation_policy'].split('.')
        if self.module.params['tiering_policy'] is not None and tier_policy[1] != self.module.params['tiering_policy']:
            modified = True
    return modified