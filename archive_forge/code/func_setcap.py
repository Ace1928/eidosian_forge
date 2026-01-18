from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
def setcap(self, path, caps):
    caps = ' '.join([''.join(cap) for cap in caps])
    cmd = "%s '%s' %s" % (self.setcap_cmd, caps, path)
    rc, stdout, stderr = self.module.run_command(cmd)
    if rc != 0:
        self.module.fail_json(msg='Unable to set capabilities of %s' % path, stdout=stdout, stderr=stderr)
    else:
        return stdout