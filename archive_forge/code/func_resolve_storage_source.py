from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, format_resource_id
def resolve_storage_source(self, source):
    blob_uri = None
    disk = None
    snapshot = None
    if isinstance(source, str) and source.lower().endswith('.vhd'):
        blob_uri = source
        return (blob_uri, disk, snapshot)
    tokenize = dict()
    if isinstance(source, dict):
        tokenize = source
    elif isinstance(source, str):
        tokenize = parse_resource_id(source)
    else:
        self.fail('source parameter should be in type string or dictionary')
    if tokenize.get('type') == 'disks':
        disk = format_resource_id(tokenize['name'], tokenize.get('subscription_id') or self.subscription_id, 'Microsoft.Compute', 'disks', tokenize.get('resource_group') or self.resource_group)
        return (blob_uri, disk, snapshot)
    if tokenize.get('type') == 'snapshots':
        snapshot = format_resource_id(tokenize['name'], tokenize.get('subscription_id') or self.subscription_id, 'Microsoft.Compute', 'snapshots', tokenize.get('resource_group') or self.resource_group)
        return (blob_uri, disk, snapshot)
    if 'type' in tokenize:
        return (blob_uri, disk, snapshot)
    snapshot_instance = self.get_snapshot(tokenize.get('resource_group') or self.resource_group, tokenize['name'])
    if snapshot_instance:
        snapshot = snapshot_instance.id
        return (blob_uri, disk, snapshot)
    disk_instance = self.get_disk(tokenize.get('resource_group') or self.resource_group, tokenize['name'])
    if disk_instance:
        disk = disk_instance.id
    return (blob_uri, disk, snapshot)