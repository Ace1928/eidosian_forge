from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def make_multi_hosts(module):
    """Create multiple hosts"""
    changed = True
    if not module.check_mode:
        hosts = []
        array = get_array(module)
        for host_num in range(module.params['start'], module.params['count'] + module.params['start']):
            if module.params['suffix']:
                hosts.append(module.params['name'] + str(host_num).zfill(module.params['digits']) + module.params['suffix'])
            else:
                hosts.append(module.params['name'] + str(host_num).zfill(module.params['digits']))
        if module.params['personality']:
            host = flasharray.HostPost(personality=module.params['personality'])
        else:
            host = flasharray.HostPost()
        res = array.post_hosts(names=hosts, host=host)
        if res.status_code != 200:
            module.fail_json(msg='Multi-Host {0}#{1} creation failed: {2}'.format(module.params['name'], module.params['suffix'], res.errors[0].message))
    module.exit_json(changed=changed)