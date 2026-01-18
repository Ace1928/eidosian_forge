from __future__ import (absolute_import, division, print_function)
from datetime import datetime, timezone
import traceback
from ansible.module_utils._text import to_native
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
def filter_replication_controller(self, replicacontrollers):

    def _deployment(obj):
        return get_deploymentconfig_for_replicationcontroller(obj) is not None

    def _zeroReplicaSize(obj):
        return obj['spec']['replicas'] == 0 and obj['status']['replicas'] == 0

    def _complete_failed(obj):
        DeploymentStatusAnnotation = 'openshift.io/deployment.phase'
        try:
            deployment_phase = obj['metadata']['annotations'].get(DeploymentStatusAnnotation)
            return deployment_phase in ('Failed', 'Complete')
        except Exception:
            return False

    def _younger(obj):
        creation_timestamp = datetime.strptime(obj['metadata']['creationTimestamp'], '%Y-%m-%dT%H:%M:%SZ')
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        age = (now - creation_timestamp).seconds / 60
        return age > self.params['keep_younger_than']

    def _orphan(obj):
        try:
            deploymentconfig_name = get_deploymentconfig_for_replicationcontroller(obj)
            params = dict(kind='DeploymentConfig', api_version='apps.openshift.io/v1', name=deploymentconfig_name, namespace=obj['metadata']['name'])
            exists = self.kubernetes_facts(**params)
            return not (exists.get['api_found'] and len(exists['resources']) > 0)
        except Exception:
            return False
    predicates = [_deployment, _zeroReplicaSize, _complete_failed]
    if self.params['orphans']:
        predicates.append(_orphan)
    if self.params['keep_younger_than']:
        predicates.append(_younger)
    results = replicacontrollers.copy()
    for pred in predicates:
        results = filter(pred, results)
    return list(results)