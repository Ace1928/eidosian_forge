from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def parseCmdOutput(self):
    patZone = 'zone name (\\S+) vsan ' + str(self.vsan)
    output = self.execute_show_zone_vsan_cmd().split('\n')
    for line in output:
        line = re.sub('[\\[].*?[\\]]', '', line)
        line = ' '.join(line.strip().split())
        if 'init' in line:
            line = line.replace('init', 'initiator')
        m = re.match(patZone, line)
        if m:
            zonename = m.group(1).strip()
            self.zDetails[zonename] = []
            continue
        elif 'pwwn' in line or 'device-alias' in line:
            v = self.zDetails[zonename]
            v.append(line)
            self.zDetails[zonename] = v