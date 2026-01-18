from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def export_keytab(module, blade):
    """Export keytab"""
    changed = False
    download_file = ''
    if blade.get_keytabs(names=[module.params['name']]).status_code == 200:
        changed = True
        if not module.check_mode:
            res = blade.get_keytabs_download(keytab_names=[module.params['name']])
            if res.status_code != 200:
                module.fail_json(msg='Failed to export keytab {0}. Error: {1}'.format(module.params['name'], res.errors[0].message))
            else:
                download_file = list(res.items)[0]
    module.exit_json(changed=changed, download_file=download_file)