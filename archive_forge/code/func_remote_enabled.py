from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_bytes, to_native
def remote_enabled(module, binary, name, method):
    """Check if the remote is enabled."""
    command = [binary, 'remote-list', '--show-disabled', '--{0}'.format(method)]
    output = _flatpak_command(module, False, command)
    for line in output.splitlines():
        listed_remote = line.split()
        if len(listed_remote) == 0:
            continue
        if listed_remote[0] == to_native(name):
            return len(listed_remote) == 1 or 'disabled' not in listed_remote[1].split(',')
    return False