from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_nas_server(self, nas_server_name, nas_server_id):
    """
        Get the NAS Server Object using NAME/ID of the NAS Server.
        :param nas_server_name: Name of the NAS Server
        :param nas_server_id: ID of the NAS Server
        :return: NAS Server object.
        """
    nas_server = nas_server_name if nas_server_name else nas_server_id
    try:
        obj_nas = self.unity_conn.get_nas_server(_id=nas_server_id, name=nas_server_name)
        if nas_server_id and obj_nas and (not obj_nas.existed):
            LOG.error('NAS Server object does not exist with ID: %s ', nas_server_id)
            return None
        return obj_nas
    except utils.HttpError as e:
        if e.http_status == 401:
            cred_err = 'Incorrect username or password , {0}'.format(e.message)
            self.module.fail_json(msg=cred_err)
        else:
            err_msg = 'Failed to get details of NAS Server {0} with error {1}'.format(nas_server, str(e))
            LOG.error(err_msg)
            self.module.fail_json(msg=err_msg)
    except utils.UnityResourceNotFoundError as e:
        err_msg = 'Failed to get details of NAS Server {0} with error {1}'.format(nas_server, str(e))
        LOG.error(err_msg)
        return None
    except Exception as e:
        nas_server = nas_server_name if nas_server_name else nas_server_id
        err_msg = 'Failed to get nas server details {0} with error {1}'.format(nas_server, str(e))
        LOG.error(err_msg)
        self.module.fail_json(msg=err_msg)