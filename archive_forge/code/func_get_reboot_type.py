from __future__ import absolute_import, division, print_function
import time
def get_reboot_type(type):
    if type == 'primary':
        return 'PrimaryNode'
    if type == 'secondary':
        return 'SecondaryNode'
    if type == 'all':
        return 'AllNodes'
    return type