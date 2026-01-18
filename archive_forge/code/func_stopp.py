from __future__ import absolute_import, division, print_function
import os
import re
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
def stopp(self):
    return self.stop()