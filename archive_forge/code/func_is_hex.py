from __future__ import absolute_import, division, print_function
import socket
from ansible.module_utils.basic import AnsibleModule
@classmethod
def is_hex(cls, number):
    try:
        int(number, 16)
    except ValueError:
        return False
    return True