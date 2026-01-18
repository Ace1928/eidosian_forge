from __future__ import (absolute_import, division, print_function)
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
def get_associated_sds(self, fault_set_id=None):
    """Get associated SDS to a fault set
            :param fault_set_id: Id of the fault set
            :return: Associated SDS details
            :rtype: dict
        """
    try:
        if fault_set_id:
            sds_details = self.powerflex_conn.fault_set.get_sdss(fault_set_id=fault_set_id)
        return sds_details
    except Exception as e:
        error_msg = f"Failed to get the associated SDS with error '{str(e)}'"
        LOG.error(error_msg)
        self.module.fail_json(msg=error_msg)