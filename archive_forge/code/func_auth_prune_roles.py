from __future__ import (absolute_import, division, print_function)
from ansible.module_utils._text import to_native
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
def auth_prune_roles(self):
    params = {'kind': 'Role', 'api_version': 'rbac.authorization.k8s.io/v1', 'namespace': self.params.get('namespace')}
    for attr in ('name', 'label_selectors'):
        if self.params.get(attr):
            params[attr] = self.params.get(attr)
    result = self.kubernetes_facts(**params)
    if not result['api_found']:
        self.fail_json(msg=result['msg'])
    roles = result.get('resources')
    if len(roles) == 0:
        self.exit_json(changed=False, msg='No candidate rolebinding to prune from namespace %s.' % self.params.get('namespace'))
    ref_roles = [(x['metadata']['namespace'], x['metadata']['name']) for x in roles]
    candidates = self.prune_resource_binding(kind='RoleBinding', api_version='rbac.authorization.k8s.io/v1', ref_kind='Role', ref_namespace_names=ref_roles, propagation_policy='Foreground')
    if len(candidates) == 0:
        self.exit_json(changed=False, role_binding=candidates)
    self.exit_json(changed=True, role_binding=candidates)