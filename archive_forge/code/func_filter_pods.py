from __future__ import absolute_import, division, print_function
import copy
import time
import traceback
from datetime import datetime
from ansible_collections.kubernetes.core.plugins.module_utils.ansiblemodule import (
from ansible_collections.kubernetes.core.plugins.module_utils.args_common import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.client import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.core import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.exceptions import (
from ansible.module_utils._text import to_native
def filter_pods(pods, force, ignore_daemonset, delete_emptydir_data):
    k8s_kind_mirror = 'kubernetes.io/config.mirror'
    daemonSet, unmanaged, mirror, localStorage, to_delete = ([], [], [], [], [])
    for pod in pods:
        if pod.metadata.annotations and k8s_kind_mirror in pod.metadata.annotations:
            mirror.append((pod.metadata.namespace, pod.metadata.name))
            continue
        if pod.status.phase in ('Succeeded', 'Failed'):
            to_delete.append((pod.metadata.namespace, pod.metadata.name))
            continue
        if pod.spec.volumes and any((vol.empty_dir for vol in pod.spec.volumes)):
            localStorage.append((pod.metadata.namespace, pod.metadata.name))
            continue
        owner_ref = pod.metadata.owner_references
        if not owner_ref:
            unmanaged.append((pod.metadata.namespace, pod.metadata.name))
        else:
            for owner in owner_ref:
                if owner.kind == 'DaemonSet':
                    daemonSet.append((pod.metadata.namespace, pod.metadata.name))
                else:
                    to_delete.append((pod.metadata.namespace, pod.metadata.name))
    warnings, errors = ([], [])
    if unmanaged:
        pod_names = ','.join([pod[0] + '/' + pod[1] for pod in unmanaged])
        if not force:
            errors.append('cannot delete Pods not managed by ReplicationController, ReplicaSet, Job, DaemonSet or StatefulSet (use option force set to yes): {0}.'.format(pod_names))
        else:
            warnings.append('Deleting Pods not managed by ReplicationController, ReplicaSet, Job, DaemonSet or StatefulSet: {0}.'.format(pod_names))
            to_delete += unmanaged
    if mirror:
        pod_names = ','.join([pod[0] + '/' + pod[1] for pod in mirror])
        warnings.append('cannot delete mirror Pods using API server: {0}.'.format(pod_names))
    if localStorage:
        pod_names = ','.join([pod[0] + '/' + pod[1] for pod in localStorage])
        if not delete_emptydir_data:
            errors.append('cannot delete Pods with local storage: {0}.'.format(pod_names))
        else:
            warnings.append('Deleting Pods with local storage: {0}.'.format(pod_names))
            for pod in localStorage:
                to_delete.append((pod[0], pod[1]))
    if daemonSet:
        pod_names = ','.join([pod[0] + '/' + pod[1] for pod in daemonSet])
        if not ignore_daemonset:
            errors.append('cannot delete DaemonSet-managed Pods (use option ignore_daemonset set to yes): {0}.'.format(pod_names))
        else:
            warnings.append('Ignoring DaemonSet-managed Pods: {0}.'.format(pod_names))
    return (to_delete, warnings, errors)