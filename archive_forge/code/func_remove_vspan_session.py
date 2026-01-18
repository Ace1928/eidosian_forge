from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def remove_vspan_session(self):
    """Calls the necessary functions to delete a VSpanSession."""
    results = dict(changed=False, result='')
    mirror_session = self.find_session_by_name()
    if mirror_session is None:
        results['result'] = 'There is no VSpanSession with the name: {0:s}.'.format(self.name)
        return results
    promiscuous_ports = self.turn_off_promiscuous()
    session_key = mirror_session.key
    self.delete_mirroring_session(session_key)
    self.deleted_session = mirror_session
    if promiscuous_ports:
        self.set_port_security_promiscuous(promiscuous_ports, True)
    results['changed'] = True
    results['result'] = 'VSpan Session has been deleted'
    return results