from __future__ import (absolute_import, division, print_function)
import os
import re
from uuid import UUID
from ansible.module_utils.six import text_type, binary_type
def rax_required_together():
    """Return the default list used for the required_together argument to
    AnsibleModule"""
    return [['api_key', 'username']]