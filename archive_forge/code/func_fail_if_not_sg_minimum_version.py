from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import missing_required_lib
def fail_if_not_sg_minimum_version(self, module_or_option, minimum_major, minimum_minor):
    version = self.get_sg_version()
    if version < (minimum_major, minimum_minor):
        msg = 'Error: ' + self.requires_sg_version(module_or_option, '%d.%d' % (minimum_major, minimum_minor))
        msg += '  Found: %s.%s.' % version
        self.module.fail_json(msg=msg)