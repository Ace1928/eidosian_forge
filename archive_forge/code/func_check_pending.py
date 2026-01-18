from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def check_pending(module):
    """Check the pending diff of the nclu buffer."""
    pending = command_helper(module, 'pending', 'Error in pending config. You may want to view `net pending` on this target.')
    delimeter1 = re.compile('net add/del commands since the last [\'"]net commit[\'"]')
    color1 = '\x1b[94m'
    if re.search(delimeter1, pending):
        pending = re.split(delimeter1, pending)[0]
        pending = pending.replace(color1, '')
    return pending.strip()