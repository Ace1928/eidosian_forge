from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible.module_utils._text import to_native
def check_vCLS_config_diff(self):
    """
        Check vCLS configuration diff
        Returns: True and all to add and to remove allowed and not allowed Datastores if there is diff, else False

        """
    if hasattr(self.cluster.configurationEx, 'systemVMsConfig'):
        vCLS_config = self.cluster.configurationEx.systemVMsConfig
    else:
        return (False, self.allowedDatastores_names, None)
    changed = False
    currentAllowedDatastores = []
    for ds in vCLS_config.allowedDatastores:
        currentAllowedDatastores.append(ds.name)
    toAddAllowedDatastores = list(set(self.allowedDatastores_names) - set(currentAllowedDatastores))
    toRemoveAllowedDatastores = list(set(currentAllowedDatastores) - set(self.allowedDatastores_names))
    if len(toAddAllowedDatastores) != 0 or len(toRemoveAllowedDatastores) != 0:
        changed = True
    return (changed, toAddAllowedDatastores, toRemoveAllowedDatastores)