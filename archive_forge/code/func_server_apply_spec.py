from __future__ import absolute_import, division, print_function
import copy
from ansible_collections.kubernetes.core.plugins.module_utils.ansiblemodule import (
from ansible_collections.kubernetes.core.plugins.module_utils.args_common import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.core import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.exceptions import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.runner import (
def server_apply_spec():
    return dict(field_manager=dict(type='str', required=True), force_conflicts=dict(type='bool', default=False))