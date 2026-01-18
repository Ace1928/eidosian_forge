from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def gluster_georep_ops(self):
    mastervol = self._validated_params('mastervol')
    slavevol = self._validated_params('slavevol')
    slavevol = self.check_pool_exclusiveness(mastervol, slavevol)
    if self.action in ['delete', 'config']:
        force = ''
    else:
        force = self._validated_params('force')
        force = 'force' if force == 'yes' else ' '
    options = 'no-verify' if self.action == 'create' else self.config_georep()
    if isinstance(options, list):
        for opt in options:
            rc, output, err = self.call_gluster_cmd('volume', 'geo-replication', mastervol, slavevol, self.action, opt, force)
    else:
        rc, output, err = self.call_gluster_cmd('volume', 'geo-replication', mastervol, slavevol, self.action, options, force)
    self._get_output(rc, output, err)
    if self.action in ['stop', 'delete'] and self.user == 'root':
        self.user = 'geoaccount'
        rc, output, err = self.call_gluster_cmd('volume', 'geo-replication', mastervol, slavevol.replace('root', 'geoaccount'), self.action, options, force)
        self._get_output(rc, output, err)