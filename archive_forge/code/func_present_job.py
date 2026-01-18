from __future__ import absolute_import, division, print_function
import os
import traceback
import xml.etree.ElementTree as ET
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
def present_job(self):
    if self.config is None and self.enabled is None:
        self.module.fail_json(msg='one of the following params is required on state=present: config,enabled')
    if not self.job_exists():
        self.create_job()
    else:
        self.update_job()