from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell\
from datetime import datetime, timedelta
import time
import copy
def modify_snap_access_mode(self, snapshot_id, snap_access_mode):
    """Modify access mode of snapshot
            :param snapshot_id: The snapshot id
            :param snap_access_mode: Access mode of the snapshot
            :return: Boolean indicating if modifying access mode of
                     snapshot is successful
        """
    try:
        self.powerflex_conn.volume.set_volume_access_mode_limit(volume_id=snapshot_id, access_mode_limit=snap_access_mode)
        return True
    except Exception as e:
        errormsg = 'Modify access mode of snapshot %s operation failed with error %s' % (snapshot_id, str(e))
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)