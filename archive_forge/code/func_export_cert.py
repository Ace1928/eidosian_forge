from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.version import (
def export_cert(module, array):
    """Export current SSL certificate"""
    changed = True
    if not module.check_mode:
        ssl = array.get_certificates(names=[module.params['name']])
        if ssl.status_code != 200:
            module.fail_json(msg='Exporting Certificate failed. Error: {0}'.format(ssl.errors[0].message))
        ssl_file = open(module.params['export_file'], 'w')
        ssl_file.write(list(ssl.items)[0].certificate)
        ssl_file.close()
    module.exit_json(changed=changed)