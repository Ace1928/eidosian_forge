from __future__ import absolute_import, division, print_function
import copy
import json
from ansible_collections.kubernetes.core.plugins.module_utils.ansiblemodule import (
from ansible_collections.kubernetes.core.plugins.module_utils.args_common import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.client import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.core import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.exceptions import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.service import (
def list_containers_in_pod(svc, resource, namespace, name):
    try:
        result = svc.client.get(resource, name=name, namespace=namespace)
        containers = [c['name'] for c in result.to_dict()['status']['containerStatuses']]
        return containers
    except Exception as exc:
        raise CoreException('Unable to retrieve log from Pod due to: {0}'.format(get_exception_message(exc)))