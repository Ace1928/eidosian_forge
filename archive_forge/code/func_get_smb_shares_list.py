from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_smb_shares_list(self):
    """Get the list of SMB shares on a given Unity storage system"""
    try:
        LOG.info('Getting SMB shares list')
        smb_shares = self.unity.get_cifs_share()
        return result_list(smb_shares)
    except Exception as e:
        msg = 'Get SMB shares from unity array failed with error %s' % str(e)
        LOG.error(msg)
        self.module.fail_json(msg=msg)