from __future__ import absolute_import, division, print_function
from ansible.errors import AnsibleFilterError
from ansible_collections.ansible.utils.plugins.module_utils.common.argspec_validate import (
def param_list_compare(*args, **kwargs):
    params = ['base', 'target']
    data = dict(zip(params, args))
    data.update(kwargs)
    if len(data) < 2:
        raise AnsibleFilterError("Missing either 'base' or 'other value in filter input,refer 'ansible.utils.param_list_compare' filter plugin documentation for details")
    valid, argspec_result, updated_params = check_argspec(DOCUMENTATION, 'param_list_compare filter', schema_conditionals=ARGSPEC_CONDITIONALS, **data)
    if not valid:
        raise AnsibleFilterError('{argspec_result} with errors: {argspec_errors}'.format(argspec_result=argspec_result.get('msg'), argspec_errors=argspec_result.get('errors')))
    base = data['base']
    other = data['target']
    combined = []
    alls = [x for x in other if x == 'all']
    bangs = [x[1:] for x in other if x.startswith('!')]
    rbangs = [x for x in other if x.startswith('!')]
    remain = [x for x in other if x not in alls and x not in rbangs and (x in base)]
    unsupported = [x for x in other if x not in alls and x not in rbangs and (x not in base)]
    if alls:
        combined = base
    for entry in bangs:
        if entry in combined:
            combined.remove(entry)
    for entry in remain:
        if entry not in combined:
            combined.append(entry)
    combined.sort()
    output = {'actionable': combined, 'unsupported': unsupported}
    return output