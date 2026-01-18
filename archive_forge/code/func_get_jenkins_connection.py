from __future__ import absolute_import, division, print_function
import traceback
from time import sleep
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
def get_jenkins_connection(self):
    try:
        if self.user and self.password:
            return jenkins.Jenkins(self.jenkins_url, self.user, self.password)
        elif self.user and self.token:
            return jenkins.Jenkins(self.jenkins_url, self.user, self.token)
        elif self.user and (not (self.password or self.token)):
            return jenkins.Jenkins(self.jenkins_url, self.user)
        else:
            return jenkins.Jenkins(self.jenkins_url)
    except Exception as e:
        self.module.fail_json(msg='Unable to connect to Jenkins server, %s' % to_native(e))