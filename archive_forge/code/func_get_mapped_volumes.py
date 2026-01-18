from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell\
def get_mapped_volumes(self, sdc_id):
    """Get volumes mapped to SDC
        :param sdc_id: The ID of the SDC
        :return: List containing volume details mapped to SDC
        """
    try:
        resp = self.powerflex_conn.sdc.get_mapped_volumes(sdc_id=sdc_id)
        return resp
    except Exception as e:
        errormsg = 'Failed to get the volumes mapped to SDC %s with error %s' % (sdc_id, str(e))
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)