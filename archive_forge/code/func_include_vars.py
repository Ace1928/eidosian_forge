import os
import testinfra.utils.ansible_runner
def include_vars(host):
    if host.system_info.distribution == 'debian' or host.system_info.distribution == 'ubuntu':
        ansible = host.ansible('include_vars', 'file="../../vars/Debian.yml"', False, False)
    else:
        ansible = host.ansible('include_vars', 'file="../../vars/RedHat.yml"', False, False)
    return ansible