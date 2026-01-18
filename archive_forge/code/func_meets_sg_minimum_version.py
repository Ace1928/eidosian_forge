from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import missing_required_lib
def meets_sg_minimum_version(self, minimum_major, minimum_minor):
    return self.get_sg_version() >= (minimum_major, minimum_minor)