from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.containers.podman.plugins.module_utils.podman.common import LooseVersion
from ansible_collections.containers.podman.plugins.module_utils.podman.common import get_podman_version
def need_update(module, executable, name, data, driver, driver_opts, debug, labels):
    cmd = [executable, 'secret', 'inspect', '--showsecret', name]
    rc, out, err = module.run_command(cmd)
    if rc != 0:
        if debug:
            module.log('PODMAN-SECRET-DEBUG: Unable to get secret info: %s' % err)
        return True
    try:
        secret = module.from_json(out)[0]
        if driver and driver != 'file' or secret['Spec']['Driver']['Name'] != 'file':
            if debug:
                module.log('PODMAN-SECRET-DEBUG: Idempotency of driver %s is not supported' % driver)
            return True
        if secret['SecretData'] != data:
            diff['after'] = '<different-secret>'
            diff['before'] = '<secret>'
            return True
        if driver_opts:
            for k, v in driver_opts.items():
                if secret['Spec']['Driver']['Options'].get(k) != v:
                    diff['after'] = '='.join([k, v])
                    diff['before'] = '='.join([k, secret['Spec']['Driver']['Options'].get(k)])
                    return True
        if labels:
            for k, v in labels.items():
                if secret['Spec']['Labels'].get(k) != v:
                    diff['after'] = '='.join([k, v])
                    diff['before'] = '='.join([k, secret['Spec']['Labels'].get(k)])
                    return True
    except Exception:
        return True
    return False