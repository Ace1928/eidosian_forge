from __future__ import absolute_import, division, print_function
import os
import tempfile
import traceback
import re
import json
import copy
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import string_types
from ansible_collections.kubernetes.core.plugins.module_utils.version import (
from ansible.module_utils.basic import AnsibleModule
def get_helm_plugin_list(self):
    """
        Return `helm plugin list`
        """
    helm_plugin_list = self.get_helm_binary() + ' plugin list'
    rc, out, err = self.run_helm_command(helm_plugin_list)
    if rc != 0 or (out == '' and err == ''):
        self.fail_json(msg='Failed to get Helm plugin info', command=helm_plugin_list, stdout=out, stderr=err, rc=rc)
    return (rc, out, err, helm_plugin_list)