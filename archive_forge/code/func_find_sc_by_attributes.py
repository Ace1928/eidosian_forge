from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def find_sc_by_attributes(module, storage_connections_service):
    for sd_conn in [sc for sc in storage_connections_service.list() if str(sc.type) == module.params['type']]:
        sd_conn_type = str(sd_conn.type)
        if sd_conn_type in ['nfs', 'posixfs', 'glusterfs', 'localfs']:
            if module.params['address'] == sd_conn.address and module.params['path'] == sd_conn.path:
                return sd_conn
        elif sd_conn_type in ['iscsi', 'fcp']:
            if module.params['address'] == sd_conn.address and module.params['target'] == sd_conn.target:
                return sd_conn