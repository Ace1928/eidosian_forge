from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell\
import copy
def modify_volume(self, vol_id, modify_dict):
    """
        Update the volume attributes
        :param vol_id: Id of the volume
        :param modify_dict: Dictionary containing the attributes of
         volume which are to be updated
        :return: True, if the operation is successful
        """
    try:
        msg = 'Dictionary containing attributes which are to be updated is {0}.'.format(str(modify_dict))
        LOG.info(msg)
        if 'auto_snap_remove_type' in modify_dict:
            snap_type = modify_dict['auto_snap_remove_type']
            msg = 'Removing/detaching the snapshot policy from a volume. auto_snap_remove_type: {0} and snapshot policy id: {1}'.format(snap_type, modify_dict['snap_pol_id'])
            LOG.info(msg)
            self.powerflex_conn.snapshot_policy.remove_source_volume(modify_dict['snap_pol_id'], vol_id, snap_type)
            msg = 'The snapshot policy has been {0}ed successfully'.format(snap_type)
            LOG.info(msg)
        if 'auto_snap_remove_type' not in modify_dict and 'snap_pol_id' in modify_dict:
            self.powerflex_conn.snapshot_policy.add_source_volume(modify_dict['snap_pol_id'], vol_id)
            msg = 'Attached the snapshot policy {0} to volume successfully.'.format(modify_dict['snap_pol_id'])
            LOG.info(msg)
        if 'new_name' in modify_dict:
            self.powerflex_conn.volume.rename(vol_id, modify_dict['new_name'])
            msg = 'The name of the volume is updated to {0} sucessfully.'.format(modify_dict['new_name'])
            LOG.info(msg)
        if 'new_size' in modify_dict:
            self.powerflex_conn.volume.extend(vol_id, modify_dict['new_size'])
            msg = 'The size of the volume is extended to {0} sucessfully.'.format(str(modify_dict['new_size']))
            LOG.info(msg)
        if 'use_rmcache' in modify_dict:
            self.powerflex_conn.volume.set_use_rmcache(vol_id, modify_dict['use_rmcache'])
            msg = 'The use RMcache is updated to {0} sucessfully.'.format(modify_dict['use_rmcache'])
            LOG.info(msg)
        if 'comp_type' in modify_dict:
            self.powerflex_conn.volume.set_compression_method(vol_id, modify_dict['comp_type'])
            msg = 'The compression method is updated to {0} successfully.'.format(modify_dict['comp_type'])
            LOG.info(msg)
        return True
    except Exception as e:
        err_msg = 'Failed to update the volume {0} with error {1}'.format(vol_id, str(e))
        LOG.error(err_msg)
        self.module.fail_json(msg=err_msg)