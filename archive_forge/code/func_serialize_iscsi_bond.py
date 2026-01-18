from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def serialize_iscsi_bond(iscsi_bonds):
    return [{'name': bond.name, 'networks': [net.name for net in bond.networks], 'storage_connections': [connection.address for connection in bond.storage_connections]} for bond in iscsi_bonds]