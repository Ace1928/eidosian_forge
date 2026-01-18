from __future__ import absolute_import, division, print_function
import traceback
from time import sleep
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
def present_build(self):
    self.build_number = self.get_next_build()
    try:
        if self.args is None:
            self.server.build_job(self.name)
        else:
            self.server.build_job(self.name, self.args)
    except Exception as e:
        self.module.fail_json(msg='Unable to create build for %s: %s' % (self.jenkins_url, to_native(e)), exception=traceback.format_exc())