from __future__ import absolute_import, division, print_function
import atexit
import time
import re
import traceback
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible.module_utils.ansible_release import __version__ as ANSIBLE_VERSION
def module_to_xapi_vm_power_state(power_state):
    """Maps module VM power states to XAPI VM power states."""
    vm_power_state_map = {'poweredon': 'running', 'poweredoff': 'halted', 'restarted': 'running', 'suspended': 'suspended', 'shutdownguest': 'halted', 'rebootguest': 'running'}
    return vm_power_state_map.get(power_state)