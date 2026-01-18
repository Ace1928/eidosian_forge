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
def parse_yaml_content(self, content):
    if not HAS_YAML:
        self.fail_json(msg=missing_required_lib('yaml'), exception=HAS_YAML)
    try:
        return list(yaml.safe_load_all(content))
    except (IOError, yaml.YAMLError) as exc:
        self.fail_json(msg='Error parsing YAML content: {0}'.format(exc), raw_data=content)