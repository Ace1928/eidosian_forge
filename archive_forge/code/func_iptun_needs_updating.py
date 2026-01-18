from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
def iptun_needs_updating(self):
    rc, out, err = self._query_iptun_props()
    NEEDS_UPDATING = False
    if rc == 0:
        configured_local, configured_remote = out.split(':')[3:]
        if self.local_address != configured_local or self.remote_address != configured_remote:
            NEEDS_UPDATING = True
        return NEEDS_UPDATING
    else:
        self.module.fail_json(msg='Failed to query tunnel interface %s properties' % self.name, err=err, rc=rc)