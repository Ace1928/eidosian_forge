from __future__ import absolute_import, division, print_function
import re  # noqa: F402
from ansible.module_utils.basic import AnsibleModule  # noqa: F402
def pod_recreate(self):
    pods = self.discover_pods()
    self.remove_associated_pods(pods)
    rc, out, err = self._command_run(self.command)
    if rc != 0:
        self.module.fail_json('Can NOT create Pod! Error: %s' % err)
    return (out, err)