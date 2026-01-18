from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell\
def rename_sdc(self, sdc_id, new_name):
    """Rename SDC
        :param sdc_id: The ID of the SDC
        :param new_name: The new name of the SDC
        :return: Boolean indicating if rename operation is successful
        """
    try:
        self.powerflex_conn.sdc.rename(sdc_id=sdc_id, name=new_name)
        return True
    except Exception as e:
        errormsg = 'Failed to rename SDC %s with error %s' % (sdc_id, str(e))
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)