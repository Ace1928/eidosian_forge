from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.six.moves import configparser, StringIO
from io import open
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def runrefreshrepo(module, auto_import_keys=False, shortname=None):
    """Forces zypper to refresh repo metadata."""
    if auto_import_keys:
        cmd = _get_cmd(module, '--gpg-auto-import-keys', 'refresh', '--force')
    else:
        cmd = _get_cmd(module, 'refresh', '--force')
    if shortname is not None:
        cmd.extend(['-r', shortname])
    rc, stdout, stderr = module.run_command(cmd, check_rc=True)
    return (rc, stdout, stderr)