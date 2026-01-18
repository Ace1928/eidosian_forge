from __future__ import (absolute_import, division, print_function)
from ansible.module_utils._text import to_native
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
def auth_prune_clusterroles(self):
    params = {'kind': 'ClusterRole', 'api_version': 'rbac.authorization.k8s.io/v1'}
    for attr in ('name', 'label_selectors'):
        if self.params.get(attr):
            params[attr] = self.params.get(attr)
    result = self.kubernetes_facts(**params)
    if not result['api_found']:
        self.fail_json(msg=result['msg'])
    clusterroles = result.get('resources')
    if len(clusterroles) == 0:
        self.exit_json(changed=False, msg='No clusterroles found matching input criteria.')
    ref_clusterroles = [(None, x['metadata']['name']) for x in clusterroles]
    candidates_cluster_binding = self.prune_resource_binding(kind='ClusterRoleBinding', api_version='rbac.authorization.k8s.io/v1', ref_kind=None, ref_namespace_names=ref_clusterroles)
    candidates_namespaced_binding = self.prune_resource_binding(kind='RoleBinding', api_version='rbac.authorization.k8s.io/v1', ref_kind='ClusterRole', ref_namespace_names=ref_clusterroles)
    self.exit_json(changed=True, cluster_role_binding=candidates_cluster_binding, role_binding=candidates_namespaced_binding)