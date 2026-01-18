from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def get_storage_connections(iscsi_bond, connection):
    resp = []
    for storage_domain_name in iscsi_bond.get('storage_domains', []):
        storage_domains_service = connection.system_service().storage_domains_service()
        storage_domain = storage_domains_service.storage_domain_service(get_id_by_name(storage_domains_service, storage_domain_name)).get()
        resp.extend(connection.follow_link(storage_domain.storage_connections))
    for storage_connection_id in iscsi_bond.get('storage_connections', []):
        resp.append(connection.system_service().storage_connections_service().storage_connection_service(storage_connection_id).get())
    return resp