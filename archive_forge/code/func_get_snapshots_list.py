from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_snapshots_list(self):
    """Get the list of snapshots on a given Unity storage system"""
    try:
        LOG.info('Getting snapshots list')
        snapshots = self.unity.get_snap()
        return result_list(snapshots)
    except Exception as e:
        msg = 'Get snapshots from unity array failed with error %s' % str(e)
        LOG.error(msg)
        self.module.fail_json(msg=msg)