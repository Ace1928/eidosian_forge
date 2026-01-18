from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six import integer_types, string_types
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
import traceback
def ensure_gitlab_package(module):
    if not HAS_GITLAB_PACKAGE:
        module.fail_json(msg=missing_required_lib('python-gitlab', url='https://python-gitlab.readthedocs.io/en/stable/'), exception=GITLAB_IMP_ERR)