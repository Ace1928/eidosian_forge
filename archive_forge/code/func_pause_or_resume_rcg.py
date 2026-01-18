from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
def pause_or_resume_rcg(self, rcg_id, rcg_details, pause, pause_mode=None):
    """Perform specified rcg action
            :param rcg_id: Unique identifier of the RCG.
            :param rcg_details: RCG details.
            :param pause: Pause or resume RCG.
            :param pause_mode: Specifies the pause mode if pause is True.
            :return: Boolean indicates if RCG action is successful
        """
    if pause and rcg_details['pauseMode'] == 'None':
        if not pause_mode:
            self.module.fail_json(msg='Specify pause_mode to perform pause on replication consistency group.')
        return self.pause(rcg_id, pause_mode)
    if not pause and (rcg_details['pauseMode'] != 'None' or rcg_details['failoverType'] in ['Failover', 'Switchover']):
        return self.resume(rcg_id)