from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
from datetime import datetime
def get_snapshot_obj(self, name=None, id=None):
    snapshot = id if id else name
    msg = 'Failed to get details of snapshot %s with error %s '
    try:
        return self.unity_conn.get_snap(name=name, _id=id)
    except utils.HttpError as e:
        if e.http_status == 401:
            cred_err = 'Incorrect username or password , {0}'.format(e.message)
            self.module.fail_json(msg=cred_err)
        else:
            err_msg = msg % (snapshot, str(e))
            LOG.error(err_msg)
            self.module.fail_json(msg=err_msg)
    except utils.UnityResourceNotFoundError as e:
        err_msg = msg % (snapshot, str(e))
        LOG.error(err_msg)
        return None
    except Exception as e:
        err_msg = msg % (snapshot, str(e))
        LOG.error(err_msg)
        self.module.fail_json(msg=err_msg)