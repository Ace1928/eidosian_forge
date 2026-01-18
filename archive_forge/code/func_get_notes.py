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
def get_notes(self, release_name):
    command = [self.get_helm_binary(), 'get', 'notes', release_name]
    rc, out, err = self.run_helm_command(' '.join(command))
    if rc != 0:
        self.fail_json(msg=err)
    return out