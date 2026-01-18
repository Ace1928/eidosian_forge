from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
def get_pause_and_freeze_value(self):
    """
        Get Pause and Freeze values
        :return: Boolean for pause and freeze
        :rtype: (bool,bool)
        """
    rcg_state = self.module.params['rcg_state']
    pause = self.module.params['pause']
    freeze = self.module.params['freeze']
    if pause is not None:
        self.module.deprecate(msg="Use 'rcg_state' param instead of 'pause'", version='3.0.0', collection_name='dellemc.powerflex')
    if freeze is not None:
        self.module.deprecate(msg="Use 'rcg_state' param instead of 'freeze'", version='3.0.0', collection_name='dellemc.powerflex')
    if rcg_state == 'pause':
        pause = True
    if rcg_state == 'resume':
        pause = False
    if rcg_state == 'freeze':
        freeze = True
    if rcg_state == 'unfreeze':
        freeze = False
    if self.module.params['pause_mode'] and (not pause):
        self.module.fail_json(msg="Specify rcg_state as 'pause' to pause replication consistency group")
    return (pause, freeze)