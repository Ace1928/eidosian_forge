from __future__ import absolute_import, division, print_function
import socket
from ansible.module_utils.basic import AnsibleModule
@classmethod
def is_valid_port(cls, port):
    return 1 <= int(port) <= 65535