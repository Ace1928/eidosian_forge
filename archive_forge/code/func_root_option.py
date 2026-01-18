from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
def root_option(root):
    if root:
        return '--root=%s' % root
    else:
        return ''