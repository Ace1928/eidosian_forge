from __future__ import absolute_import, division, print_function
import time
import re
from collections import namedtuple
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import python_2_unicode_compatible
def exit_fail(self, msg, status=None, **kwargs):
    kwargs.update({'msg': msg, 'monit_version': self._raw_version, 'process_status': str(status) if status else None})
    self.module.fail_json(**kwargs)