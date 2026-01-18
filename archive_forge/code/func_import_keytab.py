from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def import_keytab(module, blade):
    """Import keytab"""
    changed = True
    if not module.check_mode:
        if module.params['filetype'] == 'binary':
            readtype = 'rb'
        else:
            readtype = 'r'
        with open(module.params['keytab_file'], readtype) as keytab_file:
            keytab_data = keytab_file.read()
        short_name = module.params['keytab_file'].split('/')[-1]
        res = blade.post_keytabs_upload(name_prefixes=module.params['prefix'], keytab_file=(short_name, keytab_data))
        if res.status_code != 200:
            module.fail_json(msg='Failed to import keytab file {0}. Error: {1}'.format(module.params['keytab_file'], res.errors[0].message))
    module.exit_json(changed=changed)