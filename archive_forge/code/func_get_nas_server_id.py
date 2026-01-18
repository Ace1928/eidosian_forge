from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_nas_server_id(self, nas_server_name):
    """Get NAS server ID.
            :param: nas_server_name: The name of NAS server
            :return: Return NAS server ID if exists
        """
    LOG.info('Getting NAS server ID')
    try:
        obj_nas = self.unity_conn.get_nas_server(name=nas_server_name)
        return obj_nas.get_id()
    except Exception as e:
        msg = 'Failed to get details of NAS server: %s with error: %s' % (nas_server_name, str(e))
        LOG.error(msg)
        self.module.fail_json(msg=msg)