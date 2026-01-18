from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
def create_rcg(self, rcg_params):
    """Create RCG"""
    try:
        resp = None
        self.remote_powerflex_conn = utils.get_powerflex_gateway_host_connection(self.module.params['remote_peer'])
        LOG.info('Got the remote peer connection object instance')
        protection_domain_id = rcg_params['protection_domain_id']
        if rcg_params['protection_domain_name']:
            protection_domain_id = self.get_protection_domain(self.powerflex_conn, rcg_params['protection_domain_name'])['id']
        remote_protection_domain_id = rcg_params['remote_peer']['protection_domain_id']
        if rcg_params['remote_peer']['protection_domain_name']:
            remote_protection_domain_id = self.get_protection_domain(self.remote_powerflex_conn, rcg_params['remote_peer']['protection_domain_name'])['id']
        if not self.module.check_mode:
            resp = self.powerflex_conn.replication_consistency_group.create(rpo=rcg_params['rpo'], protection_domain_id=protection_domain_id, remote_protection_domain_id=remote_protection_domain_id, destination_system_id=self.remote_powerflex_conn.system.get()[0]['id'], name=rcg_params['rcg_name'], activity_mode=rcg_params['activity_mode'])
        return (True, resp)
    except Exception as e:
        errormsg = 'Create replication consistency group failed with error {0}'.format(str(e))
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)