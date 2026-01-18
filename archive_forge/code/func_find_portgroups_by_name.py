from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible.module_utils.six.moves.urllib.parse import unquote
def find_portgroups_by_name(self, content, name=None):
    vimtype = [vim.dvs.DistributedVirtualPortgroup]
    container = content.viewManager.CreateContainerView(content.rootFolder, vimtype, True)
    if not name:
        obj = container.view
    else:
        obj = []
        for c in container.view:
            if name in c.name:
                obj.append(c)
    return obj