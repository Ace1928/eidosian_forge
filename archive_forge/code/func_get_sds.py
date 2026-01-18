from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell\
def get_sds(self, sds_name=None, sds_id=None):
    """Get SDS details
            :param sds_name: Name of the SDS
            :param sds_id: ID of the SDS
            :return: SDS details
            :rtype: dict
        """
    name_or_id = sds_id if sds_id else sds_name
    try:
        sds_details = None
        if sds_id:
            sds_details = self.powerflex_conn.sds.get(filter_fields={'id': sds_id})
        if sds_name:
            sds_details = self.powerflex_conn.sds.get(filter_fields={'name': sds_name})
        if not sds_details:
            error_msg = "Unable to find the SDS with '%s'. Please enter a valid SDS name/id." % name_or_id
            LOG.error(error_msg)
            self.module.fail_json(msg=error_msg)
        return sds_details[0]
    except Exception as e:
        error_msg = "Failed to get the SDS '%s' with error '%s'" % (name_or_id, str(e))
        LOG.error(error_msg)
        self.module.fail_json(msg=error_msg)