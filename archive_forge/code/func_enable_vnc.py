from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def enable_vnc(module, array, app):
    """Enable VNC port"""
    changed = False
    vnc_fact = []
    if not app.vnc_enabled:
        changed = True
        if not module.check_mode:
            res = array.patch_apps(names=[module.params['name']], app=App(vnc_enabled=True))
            if res.status_code == 200:
                vnc_nodes = list(array.get_apps_nodes(app_names=[module.params['name']]).items)[0]
                vnc_fact = {'status': vnc_nodes.status, 'index': vnc_nodes.index, 'version': vnc_nodes.version, 'vnc': vnc_nodes.vnc, 'name': module.params['name']}
            else:
                module.fail_json(msg='Enabling VNC for {0} failed. Error: {1}'.format(module.params['name'], res.errors[0].message))
    module.exit_json(changed=changed, vnc=vnc_fact)