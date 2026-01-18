from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import env_fallback
from . import client
def get_mutation_payload(source, *wanted_params):
    payload = get_spec_payload(source, *wanted_params)
    payload['metadata'] = dict(name=source['name'])
    if 'namespace' in source:
        if not source['namespace']:
            raise AssertionError('BUG: namespace should not be None')
        payload['metadata']['namespace'] = source['namespace']
    for kind in ('labels', 'annotations'):
        if source.get(kind):
            payload['metadata'][kind] = dict(((k, str(v)) for k, v in source[kind].items()))
    return payload