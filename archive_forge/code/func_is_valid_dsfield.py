from __future__ import absolute_import, division, print_function
import socket
from ansible.module_utils.basic import AnsibleModule
@classmethod
def is_valid_dsfield(cls, dsfield):
    dsmask = None
    if dsfield.count(':') == 1:
        dsval = dsfield.split(':')[0]
    else:
        dsval, dsmask = dsfield.split(':')
    if dsmask and (not 1 <= int(dsmask, 16) <= 255) and (not 1 <= int(dsval, 16) <= 255):
        return False
    elif not 1 <= int(dsval, 16) <= 255:
        return False
    return True