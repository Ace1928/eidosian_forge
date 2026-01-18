from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def build_change_message(self, operation, changed_list):
    """Build the changed message"""
    if operation == 'add':
        changed_operation = 'added'
    elif operation == 'remove':
        changed_operation = 'removed'
    if self.module.check_mode:
        changed_suffix = ' would be %s' % changed_operation
    else:
        changed_suffix = ' %s' % changed_operation
    if len(changed_list) > 2:
        message = ', '.join(changed_list[:-1]) + ', and ' + str(changed_list[-1])
    elif len(changed_list) == 2:
        message = ' and '.join(changed_list)
    elif len(changed_list) == 1:
        message = changed_list[0]
    message += changed_suffix
    return message