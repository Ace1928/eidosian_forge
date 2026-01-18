from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
from datetime import datetime
def generate_kerb_dict(blade):
    kerb_info = {}
    keytabs = list(blade.get_keytabs().items)
    for ktab in range(0, len(keytabs)):
        keytab_name = keytabs[ktab].prefix
        kerb_info[keytab_name] = {}
        for key in range(0, len(keytabs)):
            if keytabs[key].prefix == keytab_name:
                kerb_info[keytab_name][keytabs[key].suffix] = {'fqdn': keytabs[key].fqdn, 'kvno': keytabs[key].kvno, 'principal': keytabs[key].principal, 'realm': keytabs[key].realm, 'encryption_type': keytabs[key].encryption_type}
    return kerb_info