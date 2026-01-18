from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.proxysql.plugins.module_utils.mysql import (
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_native
def perform_checks(module):
    if module.params['port'] < 0 or module.params['port'] > 65535:
        module.fail_json(msg='port must be a valid unix port number (0-65535)')
    if module.params['compression']:
        if module.params['compression'] < 0 or module.params['compression'] > 102400:
            module.fail_json(msg='compression must be set between 0 and 102400')
    if module.params['max_replication_lag']:
        if module.params['max_replication_lag'] < 0 or module.params['max_replication_lag'] > 126144000:
            module.fail_json(msg='max_replication_lag must be set between 0 and 102400')