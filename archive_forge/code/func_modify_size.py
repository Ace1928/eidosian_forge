from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell\
from datetime import datetime, timedelta
import time
import copy
def modify_size(self, snapshot_id, new_size):
    """Modify snapshot size
            :param snapshot_id: The snapshot id
            :param new_size: Size of the snapshot
            :return: Boolean indicating if extend operation is successful
        """
    try:
        self.powerflex_conn.volume.extend(snapshot_id, new_size)
        return True
    except Exception as e:
        errormsg = 'Extend snapshot %s operation failed with error %s' % (snapshot_id, str(e))
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)