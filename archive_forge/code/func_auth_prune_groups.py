from __future__ import (absolute_import, division, print_function)
from ansible.module_utils._text import to_native
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
def auth_prune_groups(self):
    groups = self.list_groups(params=self.params)
    if len(groups) == 0:
        self.exit_json(changed=False, result="No resource type 'Group' found matching input criteria.")
    names = [x['metadata']['name'] for x in groups]
    changed = False
    rolebinding, changed_role = self.update_resource_binding(ref_kind='Group', ref_names=names, namespaced=True)
    changed = changed or changed_role
    clusterrolesbinding, changed_cr = self.update_resource_binding(ref_kind='Group', ref_names=names)
    changed = changed or changed_cr
    sccs, changed_sccs = self.update_security_context(names, 'groups')
    changed = changed or changed_sccs
    self.exit_json(changed=changed, cluster_role_binding=clusterrolesbinding, role_binding=rolebinding, security_context_constraints=sccs)