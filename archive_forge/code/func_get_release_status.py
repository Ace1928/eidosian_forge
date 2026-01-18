from __future__ import absolute_import, division, print_function
import traceback
import copy
from ansible.module_utils.basic import missing_required_lib
from ansible_collections.kubernetes.core.plugins.module_utils.helm import (
from ansible_collections.kubernetes.core.plugins.module_utils.helm_args_common import (
def get_release_status(module, release_name, release_state, get_all_values=False):
    list_command = module.get_helm_binary() + ' list --output=yaml'
    valid_release_states = ['all', 'deployed', 'failed', 'pending', 'superseded', 'uninstalled', 'uninstalling']
    for local_release_state in release_state:
        if local_release_state in valid_release_states:
            list_command += ' --%s' % local_release_state
    list_command += ' --filter ' + release_name
    rc, out, err = module.run_helm_command(list_command)
    if rc != 0:
        module.fail_json(msg='Failure when executing Helm command. Exited {0}.\nstdout: {1}\nstderr: {2}'.format(rc, out, err), command=list_command)
    release = get_release(yaml.safe_load(out), release_name)
    if release is None:
        return None
    release['values'] = module.get_values(release_name, get_all_values)
    release['manifest'] = module.get_manifest(release_name)
    release['notes'] = module.get_notes(release_name)
    release['hooks'] = module.get_hooks(release_name)
    return release