from __future__ import absolute_import, division, print_function
import logging
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_volume_details(self, vol_name=None, vol_id=None):
    """Get the details of a volume.
            :param vol_name: The name of the volume
            :param vol_id: The id of the volume
            :return: Dict containing volume details if exists
        """
    id_or_name = vol_id if vol_id else vol_name
    try:
        lun = self.unity_conn.get_lun(name=vol_name, _id=vol_id)
        cg = None
        if lun.existed:
            lunid = lun.get_id()
            unitylun = utils.UnityLun.get(self.unity_conn._cli, lunid)
            if unitylun.cg is not None:
                cg = unitylun.cg
        else:
            errormsg = 'The volume {0} not found.'.format(id_or_name)
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)
        cg_details = self.get_details(cg_id=self.module.params['cg_id'], cg_name=self.module.params['cg_name'])
        if cg is None:
            return lun._get_properties()['id']
        errormsg = 'The volume {0} is already part of consistency group {1}'.format(id_or_name, cg.name)
        if cg_details is None:
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)
        if cg.id != cg_details['id']:
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)
        return lun._get_properties()['id']
    except Exception as e:
        msg = 'Failed to get the volume {0} with error {1}'.format(id_or_name, str(e))
        LOG.error(msg)
        self.module.fail_json(msg=msg)