from __future__ import (absolute_import, division, print_function)
import base64
import logging
import os
import ssl
import time
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils._text import to_native
def fail_if_not_rest_minimum_version(self, module_name, minimum_generation, minimum_major, minimum_minor=0):
    status_code = self.get_ontap_version_using_rest()
    msgs = []
    if self.use_rest == 'never':
        msgs.append('Error: REST is required for this module, found: "use_rest: %s".' % self.use_rest)
    self.use_rest = 'always'
    if self.is_rest_error:
        msgs.append('Error using REST for version, error: %s.' % self.is_rest_error)
    if status_code != 200:
        msgs.append('Error using REST for version, status_code: %s.' % status_code)
    if msgs:
        self.module.fail_json(msg='  '.join(msgs))
    version = self.get_ontap_version()
    if version < (minimum_generation, minimum_major, minimum_minor):
        msg = 'Error: ' + self.requires_ontap_version(module_name, '%d.%d.%d' % (minimum_generation, minimum_major, minimum_minor))
        msg += '  Found: %s.%s.%s.' % version
        self.module.fail_json(msg=msg)