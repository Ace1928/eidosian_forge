from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_sp_enum(destination_sp):
    """Getting correct enum values for Storage Processor
            :param: destination_sp: Storage Processor to be used in Destination NAS Server.
            :return: enum value for Storage Processor.
        """
    if utils.NodeEnum[destination_sp]:
        destination_sp_enum = utils.NodeEnum[destination_sp]
        return destination_sp_enum