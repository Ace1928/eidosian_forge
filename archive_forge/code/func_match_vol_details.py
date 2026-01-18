from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell\
from datetime import datetime, timedelta
import time
import copy
def match_vol_details(self, snapshot):
    """Match the given volume details with the response
            :param snapshot: The snapshot details
        """
    vol_name = self.module.params['vol_name']
    vol_id = self.module.params['vol_id']
    try:
        if vol_name and vol_name != snapshot['ancestorVolumeName']:
            errormsg = 'Given volume name do not match with the corresponding snapshot details.'
            self.module.fail_json(msg=errormsg)
        if vol_id and vol_id != snapshot['ancestorVolumeId']:
            errormsg = 'Given volume ID do not match with the corresponding snapshot details.'
            self.module.fail_json(msg=errormsg)
    except Exception as e:
        errormsg = 'Failed to match volume details with the snapshot with error %s' % str(e)
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)