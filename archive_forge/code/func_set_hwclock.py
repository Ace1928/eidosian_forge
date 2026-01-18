from __future__ import absolute_import, division, print_function
import errno
import os
import platform
import random
import re
import string
import filecmp
from ansible.module_utils.basic import AnsibleModule, get_distribution
from ansible.module_utils.six import iteritems
def set_hwclock(self, value):
    if value == 'local':
        option = '--localtime'
        utc = 'no'
    else:
        option = '--utc'
        utc = 'yes'
    if self.conf_files['hwclock'] is not None:
        self._edit_file(filename=self.conf_files['hwclock'], regexp=self.regexps['hwclock'], value='UTC=%s\n' % utc, key='hwclock')
    self.execute(self.update_hwclock, '--systohc', option, log=True)