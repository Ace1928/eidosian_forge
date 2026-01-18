from __future__ import absolute_import, division, print_function
import os
import json
import tempfile
from ansible.module_utils.six.moves import shlex_quote
from ansible.module_utils.six import integer_types
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def preflight_validation(bin_path, project_path, version, variables_args=None, plan_file=None):
    if project_path is None or '/' not in project_path:
        module.fail_json(msg="Path for Terraform project can not be None or ''.")
    if not os.path.exists(bin_path):
        module.fail_json(msg="Path for Terraform binary '{0}' doesn't exist on this host - check the path and try again please.".format(bin_path))
    if not os.path.isdir(project_path):
        module.fail_json(msg="Path for Terraform project '{0}' doesn't exist on this host - check the path and try again please.".format(project_path))
    if LooseVersion(version) < LooseVersion('0.15.0'):
        module.run_command([bin_path, 'validate', '-no-color'] + variables_args, check_rc=True, cwd=project_path)
    else:
        module.run_command([bin_path, 'validate', '-no-color'], check_rc=True, cwd=project_path)