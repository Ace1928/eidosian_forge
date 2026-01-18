from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def manage_images(self):
    pool = self.params['pool']
    state = self.params['state']
    if state == 'vacuumed':
        cmd = '{0} vacuum -f'.format(self.cmd)
        rc, stdout, stderr = self.module.run_command(cmd)
        if rc != 0:
            self.module.fail_json(msg='Failed to vacuum images: {0}'.format(self.errmsg(stderr)))
        elif stdout == '':
            self.changed = False
        else:
            self.changed = True
    if self.present:
        cmd = '{0} import -P {1} -q {2}'.format(self.cmd, pool, self.uuid)
        rc, stdout, stderr = self.module.run_command(cmd)
        if rc != 0:
            self.module.fail_json(msg='Failed to import image: {0}'.format(self.errmsg(stderr)))
        regex = 'Image {0} \\(.*\\) is already installed, skipping'.format(self.uuid)
        if re.match(regex, stdout):
            self.changed = False
        regex = '.*ActiveImageNotFound.*'
        if re.match(regex, stderr):
            self.changed = False
        regex = 'Imported image {0}.*'.format(self.uuid)
        if re.match(regex, stdout.splitlines()[-1]):
            self.changed = True
    else:
        cmd = '{0} delete -P {1} {2}'.format(self.cmd, pool, self.uuid)
        rc, stdout, stderr = self.module.run_command(cmd)
        regex = '.*ImageNotInstalled.*'
        if re.match(regex, stderr):
            self.changed = False
        regex = 'Deleted image {0}'.format(self.uuid)
        if re.match(regex, stdout):
            self.changed = True