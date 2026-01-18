from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def record_to_dict(self, record):
    record_type = record.type[len('Microsoft.Network/dnszones/'):]
    records = getattr(record, RECORDSET_VALUE_MAP.get(record_type))
    if records:
        if not isinstance(records, list):
            records = [records]
    else:
        records = []
    return dict(id=record.id, relative_name=record.name, record_type=record_type, records=[x.as_dict() for x in records], time_to_live=record.ttl, fqdn=record.fqdn, provisioning_state=record.provisioning_state, metadata=record.metadata)