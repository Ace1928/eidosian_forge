from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.infinidat.infinibox.plugins.module_utils.infinibox import (
def get_sys_host(module):
    """ Get parameters """
    system = get_system(module)
    host = get_host(module, system)
    return (system, host)