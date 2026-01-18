from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_smb_share_obj(self, share_id=None, share_name=None, filesystem_obj=None, snap_obj=None, nas_obj=None):
    """Get SMB share details"""
    msg = 'Failed to get details of SMB Share {0} with error {1} '
    smb_share = share_name if share_name else share_id
    try:
        if share_id:
            obj_smb = self.unity_conn.get_cifs_share(_id=share_id)
            if obj_smb and obj_smb.existed:
                LOG.info('Successfully got the SMB share object %s ', obj_smb)
                return obj_smb
        elif share_name is not None and filesystem_obj:
            return self.unity_conn.get_cifs_share(name=share_name, filesystem=filesystem_obj)
        elif share_name is not None and snap_obj:
            return self.unity_conn.get_cifs_share(name=share_name, snap=snap_obj)
        elif share_name is not None and nas_obj:
            smb_share_obj = self.unity_conn.get_cifs_share(name=share_name)
            if isinstance(smb_share_obj, utils.cifs_share.UnityCifsShareList):
                LOG.info('Multiple SMB share with same name found.')
                smb_share_obj_list = smb_share_obj
                for smb_share in smb_share_obj_list:
                    if smb_share.filesystem.nas_server == nas_obj:
                        return smb_share
                msg = 'No SMB share found with the given NAS Server. Please provide correct share name and nas server details.'
                return None
            if smb_share_obj.filesystem.nas_server == nas_obj:
                return smb_share_obj
            msg = 'No SMB share found with the given NAS Server. Please provide correct share name and nas server details.'
            return None
        else:
            self.module.fail_json(msg='Share Name is Passed. Please enter Filesystem/Snapshot/NAS Server Resource along with share_name to get the details of the SMB share')
    except utils.HttpError as e:
        if e.http_status == 401:
            cred_err = 'Incorrect username or password , {0}'.format(e.message)
            self.module.fail_json(msg=cred_err)
        else:
            err_msg = msg.format(smb_share, str(e))
            LOG.error(err_msg)
            self.module.fail_json(msg=err_msg)
    except utils.UnityResourceNotFoundError as e:
        err_msg = msg.format(smb_share, str(e))
        LOG.error(err_msg)
        return None
    except Exception as e:
        err_msg = msg.format(smb_share, str(e))
        LOG.error(err_msg)
        self.module.fail_json(msg=err_msg)