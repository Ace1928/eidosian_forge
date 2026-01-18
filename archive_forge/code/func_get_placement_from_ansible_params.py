from __future__ import absolute_import, division, print_function
import shlex
import time
import traceback
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils.common import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible.module_utils.basic import human_to_bytes
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_text, to_native
@staticmethod
def get_placement_from_ansible_params(params):
    placement = params['placement'] or {}
    constraints = get_value('constraints', placement)
    preferences = placement.get('preferences')
    replicas_max_per_node = get_value('replicas_max_per_node', placement)
    return {'constraints': constraints, 'placement_preferences': preferences, 'replicas_max_per_node': replicas_max_per_node}