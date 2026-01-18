from __future__ import absolute_import, division, print_function
import os
import traceback
import xml.etree.ElementTree as ET
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
def job_exists(self):
    try:
        return bool(self.server.job_exists(self.name))
    except Exception as e:
        self.module.fail_json(msg='Unable to validate if job exists, %s for %s' % (to_native(e), self.jenkins_url), exception=traceback.format_exc())