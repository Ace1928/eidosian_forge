from __future__ import absolute_import, division, print_function
import traceback
from time import sleep
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
def get_next_build(self):
    try:
        build_number = self.server.get_job_info(self.name)['nextBuildNumber']
    except Exception as e:
        self.module.fail_json(msg='Unable to get job info from Jenkins server, %s' % to_native(e), exception=traceback.format_exc())
    return build_number