from __future__ import (absolute_import, division, print_function)
from ansible.module_utils._text import to_native
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
def auth_prune_users(self):
    params = {'kind': 'User', 'api_version': 'user.openshift.io/v1'}
    for attr in ('name', 'label_selectors'):
        if self.params.get(attr):
            params[attr] = self.params.get(attr)
    users = self.kubernetes_facts(**params)
    if len(users) == 0:
        self.exit_json(changed=False, msg="No resource type 'User' found matching input criteria.")
    names = [x['metadata']['name'] for x in users]
    changed = False
    rolebinding, changed_role = self.update_resource_binding(ref_kind='User', ref_names=names, namespaced=True)
    changed = changed or changed_role
    clusterrolesbinding, changed_cr = self.update_resource_binding(ref_kind='User', ref_names=names)
    changed = changed or changed_cr
    sccs, changed_sccs = self.update_security_context(names, 'users')
    changed = changed or changed_sccs
    groups = self.list_groups()
    deleted_groups = []
    resource = self.find_resource(kind='Group', api_version='user.openshift.io/v1')
    for grp in groups:
        subjects = grp.get('users', [])
        retainedSubjects = [x for x in subjects if x not in names]
        if len(subjects) != len(retainedSubjects):
            deleted_groups.append(grp['metadata']['name'])
            changed = True
            if not self.check_mode:
                upd_group = grp
                upd_group.update({'users': retainedSubjects})
                try:
                    resource.apply(upd_group, namespace=None)
                except DynamicApiError as exc:
                    msg = 'Failed to apply object due to: {0}'.format(exc.body)
                    self.fail_json(msg=msg)
    oauth = self.kubernetes_facts(kind='OAuthClientAuthorization', api_version='oauth.openshift.io/v1')
    deleted_auths = []
    resource = self.find_resource(kind='OAuthClientAuthorization', api_version='oauth.openshift.io/v1')
    for authorization in oauth:
        if authorization.get('userName', None) in names:
            auth_name = authorization['metadata']['name']
            deleted_auths.append(auth_name)
            changed = True
            if not self.check_mode:
                try:
                    resource.delete(name=auth_name, namespace=None, body=client.V1DeleteOptions())
                except DynamicApiError as exc:
                    msg = 'Failed to delete OAuthClientAuthorization {name} due to: {msg}'.format(name=auth_name, msg=exc.body)
                    self.fail_json(msg=msg)
                except Exception as e:
                    msg = 'Failed to delete OAuthClientAuthorization {name} due to: {msg}'.format(name=auth_name, msg=to_native(e))
                    self.fail_json(msg=msg)
    self.exit_json(changed=changed, cluster_role_binding=clusterrolesbinding, role_binding=rolebinding, security_context_constraints=sccs, authorization=deleted_auths, group=deleted_groups)