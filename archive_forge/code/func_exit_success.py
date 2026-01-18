from __future__ import absolute_import, division, print_function
import time
import re
from collections import namedtuple
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import python_2_unicode_compatible
def exit_success(self, state):
    self.module.exit_json(changed=True, name=self.process_name, state=state)