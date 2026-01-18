from __future__ import absolute_import, division, print_function
import re
import tempfile
import traceback
import copy
from ansible_collections.kubernetes.core.plugins.module_utils.version import (
from ansible.module_utils.basic import missing_required_lib
from ansible_collections.kubernetes.core.plugins.module_utils.helm import (
from ansible_collections.kubernetes.core.plugins.module_utils.helm_args_common import (
def get_plugin_version(plugin):
    """
    Check if helm plugin is installed and return corresponding version
    """
    rc, output, err, command = module.get_helm_plugin_list()
    out = parse_helm_plugin_list(output=output.splitlines())
    if not out:
        return None
    for line in out:
        if line[0] == plugin:
            return line[1]
    return None