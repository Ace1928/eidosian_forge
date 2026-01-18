from __future__ import absolute_import, division, print_function
import re
import socket
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
def start_volume(name):
    run_gluster(['volume', 'start', name])