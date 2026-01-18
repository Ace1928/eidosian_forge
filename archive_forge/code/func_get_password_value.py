from __future__ import absolute_import, division, print_function
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.basic import AnsibleModule
def get_password_value(module, pkg, question, vtype):
    getsel = module.get_bin_path('debconf-get-selections', True)
    cmd = [getsel]
    rc, out, err = module.run_command(cmd)
    if rc != 0:
        module.fail_json(msg="Failed to get the value '%s' from '%s'" % (question, pkg))
    desired_line = None
    for line in out.split('\n'):
        if line.startswith(pkg):
            desired_line = line
            break
    if not desired_line:
        module.fail_json(msg="Failed to find the value '%s' from '%s'" % (question, pkg))
    dpkg, dquestion, dvtype, dvalue = desired_line.split()
    if dquestion == question and dvtype == vtype:
        return dvalue
    return ''