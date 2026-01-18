from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell\
def get_sdc(self, sdc_name=None, sdc_ip=None, sdc_id=None):
    """Get the SDC Details
            :param sdc_name: The name of the SDC
            :param sdc_ip: The IP of the SDC
            :param sdc_id: The ID of the SDC
            :return: The dict containing SDC details
        """
    if sdc_name:
        id_ip_name = sdc_name
    elif sdc_ip:
        id_ip_name = sdc_ip
    else:
        id_ip_name = sdc_id
    try:
        if sdc_name:
            sdc_details = self.powerflex_conn.sdc.get(filter_fields={'name': sdc_name})
        elif sdc_ip:
            sdc_details = self.powerflex_conn.sdc.get(filter_fields={'sdcIp': sdc_ip})
        else:
            sdc_details = self.powerflex_conn.sdc.get(filter_fields={'id': sdc_id})
        if len(sdc_details) == 0:
            error_msg = 'Unable to find SDC with identifier %s' % id_ip_name
            LOG.error(error_msg)
            return None
        sdc_details[0]['mapped_volumes'] = self.get_mapped_volumes(sdc_details[0]['id'])
        return sdc_details[0]
    except Exception as e:
        errormsg = 'Failed to get the SDC %s with error %s' % (id_ip_name, str(e))
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)