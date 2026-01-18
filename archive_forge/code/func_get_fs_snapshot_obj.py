from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
from datetime import datetime
def get_fs_snapshot_obj(self, name=None, id=None):
    fs_snapshot = id if id else name
    msg = 'Failed to get details of filesystem snapshot %s with error %s.'
    try:
        fs_snap_obj = self.unity_conn.get_snap(name=name, _id=id)
        if fs_snap_obj and fs_snap_obj.existed:
            LOG.info('Successfully got the filesystem snapshot object %s.', fs_snap_obj)
        else:
            fs_snap_obj = None
        return fs_snap_obj
    except utils.HttpError as e:
        if e.http_status == 401:
            cred_err = 'Incorrect username or password , %s' % e.message
            self.module.fail_json(msg=cred_err)
        else:
            err_msg = msg % (fs_snapshot, str(e))
            LOG.error(err_msg)
            self.module.fail_json(msg=err_msg)
    except utils.UnityResourceNotFoundError as e:
        err_msg = msg % (fs_snapshot, str(e))
        LOG.error(err_msg)
        return None
    except Exception as e:
        err_msg = msg % (fs_snapshot, str(e))
        LOG.error(err_msg)
        self.module.fail_json(msg=err_msg)