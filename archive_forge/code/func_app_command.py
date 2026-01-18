from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import os
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
import platform
def app_command(self, command, id):
    """ Runs a `mas` command on a given app; command can be 'install', 'upgrade' or 'uninstall' """
    if not self.module.check_mode:
        if command != 'uninstall':
            self.check_signin()
        rc, out, err = self.run([command, str(id)])
        if rc != 0:
            self.module.fail_json(msg="Error running command '{0}' on app '{1}': {2}".format(command, str(id), out.rstrip()))
    self.__dict__['count_' + command] += 1