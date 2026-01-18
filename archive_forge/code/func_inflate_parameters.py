from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _snake_to_camel
from ansible.module_utils.six import string_types
def inflate_parameters(self, spec, body, level):
    if isinstance(body, list):
        for item in body:
            self.inflate_parameters(spec, item, level)
        return
    for name in spec.keys():
        param = body.get(name)
        if param is None:
            if spec[name].get('purgeIfNone', False):
                body.pop(name, None)
            continue
        pattern = spec[name].get('pattern', None)
        if pattern:
            if pattern == 'camelize':
                param = _snake_to_camel(param, True)
            elif isinstance(pattern, list):
                normalized = None
                for p in pattern:
                    normalized = self.normalize_resource_id(param, p)
                    body[name] = normalized
                    if normalized is not None:
                        break
            else:
                param = self.normalize_resource_id(param, pattern)
                body[name] = param
        disposition = spec[name].get('disposition', '*')
        if level == 0 and (not disposition.startswith('/')):
            continue
        if disposition == '/':
            disposition = '/*'
        parts = disposition.split('/')
        if parts[0] == '':
            parts.pop(0)
        target_dict = body
        elem = body.pop(name)
        while len(parts) > 1:
            target_dict = target_dict.setdefault(parts.pop(0), {})
        targetName = parts[0] if parts[0] != '*' else name
        target_dict[targetName] = elem
        if spec[name].get('options'):
            self.inflate_parameters(spec[name].get('options'), target_dict[targetName], level + 1)