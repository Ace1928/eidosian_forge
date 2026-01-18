from __future__ import absolute_import, division, print_function
import time
import re
from collections import namedtuple
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import python_2_unicode_compatible
def is_process_present(self):
    command = [self.monit_bin_path, 'summary'] + self.command_args
    rc, out, err = self.module.run_command(command, check_rc=True)
    return bool(re.findall('\\b%s\\b' % self.process_name, out))