from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
def query_package_provides(module, name, root):
    rpm_path = module.get_bin_path('rpm', True)
    cmd = '%s -q --whatprovides %s %s' % (rpm_path, name, root_option(root))
    rc, stdout, stderr = module.run_command(cmd, check_rc=False)
    return rc == 0