from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def deactivate_ports(module, port_path, ports, stdout, stderr):
    """ Deactivate a port if it's active. """
    deactivated_c = 0
    for port in ports:
        if not query_port(module, port_path, port):
            module.fail_json(msg='Failed to deactivate %s, port(s) not present' % port, stdout=stdout, stderr=stderr)
        if not query_port(module, port_path, port, state='active'):
            continue
        rc, out, err = module.run_command('%s deactivate %s' % (port_path, port))
        stdout += out
        stderr += err
        if query_port(module, port_path, port, state='active'):
            module.fail_json(msg='Failed to deactivate %s: %s' % (port, err), stdout=stdout, stderr=stderr)
        deactivated_c += 1
    if deactivated_c > 0:
        module.exit_json(changed=True, msg='Deactivated %s port(s)' % deactivated_c, stdout=stdout, stderr=stderr)
    module.exit_json(changed=False, msg='Port(s) already inactive', stdout=stdout, stderr=stderr)