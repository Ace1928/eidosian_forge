from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def install_flat(module, binary, remote, names, method, no_dependencies):
    """Add new flatpaks."""
    global result
    uri_names = []
    id_names = []
    for name in names:
        if name.startswith('http://') or name.startswith('https://'):
            uri_names.append(name)
        else:
            id_names.append(name)
    base_command = [binary, 'install', '--{0}'.format(method)]
    flatpak_version = _flatpak_version(module, binary)
    if LooseVersion(flatpak_version) < LooseVersion('1.1.3'):
        base_command += ['-y']
    else:
        base_command += ['--noninteractive']
    if no_dependencies:
        base_command += ['--no-deps']
    if uri_names:
        command = base_command + uri_names
        _flatpak_command(module, module.check_mode, command)
    if id_names:
        command = base_command + [remote] + id_names
        _flatpak_command(module, module.check_mode, command)
    result['changed'] = True