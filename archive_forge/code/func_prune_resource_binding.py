from __future__ import (absolute_import, division, print_function)
from ansible.module_utils._text import to_native
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
def prune_resource_binding(self, kind, api_version, ref_kind, ref_namespace_names, propagation_policy=None):
    resource = self.find_resource(kind=kind, api_version=api_version, fail=True)
    candidates = []
    for ref_namespace, ref_name in ref_namespace_names:
        try:
            result = resource.get(name=None, namespace=ref_namespace)
            result = result.to_dict()
            result = result.get('items') if 'items' in result else [result]
            for obj in result:
                namespace = obj['metadata'].get('namespace', None)
                name = obj['metadata'].get('name')
                if ref_kind and obj['roleRef']['kind'] != ref_kind:
                    continue
                if obj['roleRef']['name'] == ref_name:
                    candidates.append((namespace, name))
        except NotFoundError:
            continue
        except DynamicApiError as exc:
            msg = 'Failed to get {kind} resource due to: {msg}'.format(kind=kind, msg=exc.body)
            self.fail_json(msg=msg)
        except Exception as e:
            msg = 'Failed to get {kind} due to: {msg}'.format(kind=kind, msg=to_native(e))
            self.fail_json(msg=msg)
    if len(candidates) == 0 or self.check_mode:
        return [y if x is None else x + '/' + y for x, y in candidates]
    delete_options = client.V1DeleteOptions()
    if propagation_policy:
        delete_options.propagation_policy = propagation_policy
    for namespace, name in candidates:
        try:
            result = resource.delete(name=name, namespace=namespace, body=delete_options)
        except DynamicApiError as exc:
            msg = 'Failed to delete {kind} {namespace}/{name} due to: {msg}'.format(kind=kind, namespace=namespace, name=name, msg=exc.body)
            self.fail_json(msg=msg)
        except Exception as e:
            msg = 'Failed to delete {kind} {namespace}/{name} due to: {msg}'.format(kind=kind, namespace=namespace, name=name, msg=to_native(e))
            self.fail_json(msg=msg)
    return [y if x is None else x + '/' + y for x, y in candidates]