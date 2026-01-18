from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
def get_publishers(module):
    rc, out, err = module.run_command(['pkg', 'publisher', '-Ftsv'], True)
    lines = out.splitlines()
    keys = lines.pop(0).lower().split('\t')
    publishers = {}
    for line in lines:
        values = dict(zip(keys, map(unstringify, line.split('\t'))))
        name = values['publisher']
        if name not in publishers:
            publishers[name] = dict(((k, values[k]) for k in ['sticky', 'enabled']))
            publishers[name]['origin'] = []
            publishers[name]['mirror'] = []
        if values['type'] is not None:
            publishers[name][values['type']].append(values['uri'])
    return publishers