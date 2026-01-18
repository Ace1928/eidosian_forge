from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.containers.podman.plugins.module_utils.podman.common import LooseVersion
from ansible_collections.containers.podman.plugins.module_utils.podman.common import get_podman_version
def podman_secret_create(module, executable, name, data, force, skip, driver, driver_opts, debug, labels):
    podman_version = get_podman_version(module, fail=False)
    if podman_version is not None and LooseVersion(podman_version) >= LooseVersion('4.7.0') and (driver is None or driver == 'file'):
        if not skip and need_update(module, executable, name, data, driver, driver_opts, debug, labels):
            podman_secret_remove(module, executable, name)
        else:
            return {'changed': False}
    else:
        if force:
            podman_secret_remove(module, executable, name)
        if skip and podman_secret_exists(module, executable, name, podman_version):
            return {'changed': False}
    cmd = [executable, 'secret', 'create']
    if driver:
        cmd.append('--driver')
        cmd.append(driver)
    if driver_opts:
        cmd.append('--driver-opts')
        cmd.append(','.join(('='.join(i) for i in driver_opts.items())))
    if labels:
        for k, v in labels.items():
            cmd.append('--label')
            cmd.append('='.join([k, v]))
    cmd.append(name)
    cmd.append('-')
    rc, out, err = module.run_command(cmd, data=data, binary_data=True)
    if rc != 0:
        module.fail_json(msg='Unable to create secret: %s' % err)
    return {'changed': True, 'diff': {'before': diff['before'] + '\n', 'after': diff['after'] + '\n'}}