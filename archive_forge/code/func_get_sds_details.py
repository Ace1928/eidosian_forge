from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell\
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.powerflex_base \
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.configuration \
import copy
def get_sds_details(self, sds_name=None, sds_id=None):
    """Get SDS details
            :param sds_name: Name of the SDS
            :type sds_name: str
            :param sds_id: ID of the SDS
            :type sds_id: str
            :return: Details of SDS if it exist
            :rtype: dict
        """
    id_or_name = sds_id if sds_id else sds_name
    try:
        if sds_name:
            sds_details = self.powerflex_conn.sds.get(filter_fields={'name': sds_name})
        else:
            sds_details = self.powerflex_conn.sds.get(filter_fields={'id': sds_id})
        if len(sds_details) == 0:
            msg = "SDS with identifier '%s' not found" % id_or_name
            LOG.info(msg)
            return None
        return sds_details[0]
    except Exception as e:
        error_msg = "Failed to get the SDS '%s' with error '%s'" % (id_or_name, str(e))
        LOG.error(error_msg)
        self.module.fail_json(msg=error_msg)