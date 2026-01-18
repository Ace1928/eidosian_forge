from __future__ import absolute_import, division, print_function
import os
import platform
import re
import tempfile
import time
from ansible.module_utils.basic import AnsibleModule
def boot(self):
    if not self.module.check_mode:
        cmd = '%s -z %s boot' % (self.zoneadm_cmd, self.name)
        rc, out, err = self.module.run_command(cmd)
        if rc != 0:
            self.module.fail_json(msg='Failed to boot zone. %s' % (out + err))
        "\n            The boot command can return before the zone has fully booted. This is especially\n            true on the first boot when the zone initializes the SMF services. Unless the zone\n            has fully booted, subsequent tasks in the playbook may fail as services aren't running yet.\n            Wait until the zone's console login is running; once that's running, consider the zone booted.\n            "
        elapsed = 0
        while True:
            if elapsed > self.timeout:
                self.module.fail_json(msg='timed out waiting for zone to boot')
            rc = os.system('ps -z %s -o args|grep "ttymon.*-d /dev/console" > /dev/null 2>/dev/null' % self.name)
            if rc == 0:
                break
            time.sleep(10)
            elapsed += 10
    self.changed = True
    self.msg.append('zone booted')