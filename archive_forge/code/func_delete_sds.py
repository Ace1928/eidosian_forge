from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell\
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.powerflex_base \
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.configuration \
import copy
def delete_sds(self, sds_id):
    """Delete SDS
            :param sds_id: SDS ID
            :type sds_id: str
            :return: Boolean indicating if delete operation is successful
        """
    try:
        if not self.module.check_mode:
            self.powerflex_conn.sds.delete(sds_id)
            return None
        return self.get_sds_details(sds_id=sds_id)
    except Exception as e:
        error_msg = "Delete SDS '%s' operation failed with error '%s'" % (sds_id, str(e))
        LOG.error(error_msg)
        self.module.fail_json(msg=error_msg)