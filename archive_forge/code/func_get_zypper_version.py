from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.six.moves import configparser, StringIO
from io import open
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def get_zypper_version(module):
    rc, stdout, stderr = module.run_command([module.get_bin_path('zypper', required=True), '--version'])
    if rc != 0 or not stdout.startswith('zypper '):
        return LooseVersion('1.0')
    return LooseVersion(stdout.split()[1])