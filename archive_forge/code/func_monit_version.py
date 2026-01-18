from __future__ import absolute_import, division, print_function
import time
import re
from collections import namedtuple
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import python_2_unicode_compatible
def monit_version(self):
    if self._monit_version is None:
        self._raw_version, version = self._get_monit_version()
        self._monit_version = (version[0], version[1])
    return self._monit_version