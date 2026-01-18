from __future__ import absolute_import, division, print_function
import socket
from ansible.module_utils.basic import AnsibleModule
def flow_needs_udpating(self):
    rc, out, err = self._query_flow_props()
    NEEDS_UPDATING = False
    if rc == 0:
        properties = (line.split(':') for line in out.rstrip().split('\n'))
        for prop, value in properties:
            if prop == 'maxbw' and self.maxbw != value:
                self._needs_updating.update({prop: True})
                NEEDS_UPDATING = True
            elif prop == 'priority' and self.priority != value:
                self._needs_updating.update({prop: True})
                NEEDS_UPDATING = True
        return NEEDS_UPDATING
    else:
        self.module.fail_json(msg='Error while checking flow properties: %s' % err, stderr=err, rc=rc)