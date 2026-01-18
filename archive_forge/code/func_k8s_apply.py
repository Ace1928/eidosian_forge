from __future__ import absolute_import, division, print_function
from collections import OrderedDict
import json
from ansible.module_utils.common.dict_transformations import dict_merge
from ansible_collections.kubernetes.core.plugins.module_utils.exceptions import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.core import (
from ansible_collections.kubernetes.core.plugins.module_utils.version import (
def k8s_apply(resource, definition, **kwargs):
    existing, desired = apply_object(resource, definition)
    server_side = kwargs.get('server_side', False)
    if server_side:
        versions = gather_versions()
        body = definition
        if LooseVersion(versions['kubernetes']) < LooseVersion('25.0.0'):
            body = json.dumps(definition).encode()
        return resource.server_side_apply(body=body, name=definition['metadata']['name'], namespace=definition['metadata'].get('namespace'), force_conflicts=kwargs.get('force_conflicts'), field_manager=kwargs.get('field_manager'), dry_run=kwargs.get('dry_run'))
    if not existing:
        return resource.create(body=desired, namespace=definition['metadata'].get('namespace'), **kwargs)
    if existing == desired:
        return resource.get(name=definition['metadata']['name'], namespace=definition['metadata'].get('namespace'))
    return resource.patch(body=desired, name=definition['metadata']['name'], namespace=definition['metadata'].get('namespace'), content_type='application/merge-patch+json', **kwargs)