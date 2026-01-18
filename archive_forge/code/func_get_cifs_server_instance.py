from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell import utils
def get_cifs_server_instance(self, cifs_server_id):
    """Get CIFS server instance.
            :param: cifs_server_id: The ID of the CIFS server
            :return: Return CIFS server instance if exists
        """
    try:
        cifs_server_obj = utils.UnityCifsServer.get(cli=self.unity_conn._cli, _id=cifs_server_id)
        return cifs_server_obj
    except Exception as e:
        error_msg = 'Failed to get the CIFS server %s instance with error %s' % (cifs_server_id, str(e))
        LOG.error(error_msg)
        self.module.fail_json(msg=error_msg)