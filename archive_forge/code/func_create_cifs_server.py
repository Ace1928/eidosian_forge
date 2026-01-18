from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell import utils
def create_cifs_server(self, nas_server_id, interfaces=None, netbios_name=None, cifs_server_name=None, domain=None, domain_username=None, domain_password=None, workgroup=None, local_password=None):
    """Create CIFS server.
            :param: nas_server_id: The ID of NAS server
            :param: interfaces: List of file interfaces
            :param: netbios_name: Name of the SMB server in windows network
            :param: cifs_server_name: Name of the CIFS server
            :param: domain: The domain name where the SMB server is registered in Active Directory
            :param: domain_username: The domain username
            :param: domain_password: The domain password
            :param: workgroup: Standalone SMB server workgroup
            :param: local_password: Standalone SMB server admin password
            :return: Return True if CIFS server is created
        """
    LOG.info('Creating CIFS server')
    try:
        if not self.module.check_mode:
            utils.UnityCifsServer.create(cli=self.unity_conn._cli, nas_server=nas_server_id, interfaces=interfaces, netbios_name=netbios_name, name=cifs_server_name, domain=domain, domain_username=domain_username, domain_password=domain_password, workgroup=workgroup, local_password=local_password)
        return True
    except Exception as e:
        msg = 'Failed to create CIFS server with error: %s' % str(e)
        LOG.error(msg)
        self.module.fail_json(msg=msg)