from __future__ import absolute_import, division, print_function
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.basic import AnsibleModule
def get_selections(module, pkg):
    cmd = [module.get_bin_path('debconf-show', True), pkg]
    rc, out, err = module.run_command(' '.join(cmd))
    if rc != 0:
        module.fail_json(msg=err)
    selections = {}
    for line in out.splitlines():
        key, value = line.split(':', 1)
        selections[key.strip('*').strip()] = value.strip()
    return selections