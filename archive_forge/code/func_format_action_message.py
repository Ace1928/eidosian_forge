from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def format_action_message(module, action, count):
    vars = {'actioned': action, 'count': count}
    if module.check_mode:
        message = 'would have %(actioned)s %(count)d package' % vars
    else:
        message = '%(actioned)s %(count)d package' % vars
    if count == 1:
        return message
    else:
        return message + 's'