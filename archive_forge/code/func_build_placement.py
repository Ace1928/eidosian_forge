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
def build_placement(self):
    placement_args = {}
    if self.constraints is not None:
        placement_args['constraints'] = self.constraints
    if self.replicas_max_per_node is not None:
        placement_args['maxreplicas'] = self.replicas_max_per_node
    if self.placement_preferences is not None:
        placement_args['preferences'] = [{key.title(): {'SpreadDescriptor': value}} for preference in self.placement_preferences for key, value in preference.items()]
    return types.Placement(**placement_args) if placement_args else None