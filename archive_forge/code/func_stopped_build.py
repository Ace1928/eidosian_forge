from __future__ import absolute_import, division, print_function
import traceback
from time import sleep
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
def stopped_build(self):
    build_info = None
    try:
        build_info = self.server.get_build_info(self.name, self.build_number)
        if build_info['building'] is True:
            self.server.stop_build(self.name, self.build_number)
    except Exception as e:
        self.module.fail_json(msg='Unable to stop build for %s: %s' % (self.jenkins_url, to_native(e)), exception=traceback.format_exc())
    else:
        if build_info['building'] is False:
            self.module.exit_json(**self.result)