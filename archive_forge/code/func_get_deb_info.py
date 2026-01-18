from __future__ import absolute_import, division, print_function
import os
import traceback
from ansible.module_utils._text import to_bytes, to_native
from ansible_collections.theforeman.foreman.plugins.module_utils.foreman_helper import KatelloAnsibleModule, missing_required_lib
def get_deb_info(path):
    control = debfile.DebFile(path).debcontrol()
    return (control['package'], control['version'], control['architecture'])