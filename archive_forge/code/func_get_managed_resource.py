from __future__ import absolute_import, division, print_function
import copy
from ansible_collections.kubernetes.core.plugins.module_utils.ansiblemodule import (
from ansible_collections.kubernetes.core.plugins.module_utils.args_common import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.client import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.core import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.exceptions import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.service import (
def get_managed_resource(kind):
    managed_resource = {}
    if kind == 'DaemonSet':
        managed_resource['kind'] = 'ControllerRevision'
        managed_resource['api_version'] = 'apps/v1'
    elif kind == 'Deployment':
        managed_resource['kind'] = 'ReplicaSet'
        managed_resource['api_version'] = 'apps/v1'
    else:
        raise CoreException('Cannot perform rollback on resource of kind {0}'.format(kind))
    return managed_resource