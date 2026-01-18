from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.containers.podman.plugins.module_utils.podman.common import LooseVersion
from ansible_collections.containers.podman.plugins.module_utils.podman.common import get_podman_version
def podman_secret_exists(module, executable, name, version):
    if version is None or LooseVersion(version) < LooseVersion('4.5.0'):
        rc, out, err = module.run_command([executable, 'secret', 'ls', '--format', '{{.Name}}'])
        return name in [i.strip() for i in out.splitlines()]
    rc, out, err = module.run_command([executable, 'secret', 'exists', name])
    return rc == 0