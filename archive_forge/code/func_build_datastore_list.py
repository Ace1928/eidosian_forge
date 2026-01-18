from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient
def build_datastore_list(self, datastore_list):
    """ Build list with datastores """
    datastores = list()
    for datastore in datastore_list:
        if self.schema == 'summary':
            summary = datastore.summary
            datastore_summary = dict()
            datastore_summary['accessible'] = summary.accessible
            datastore_summary['capacity'] = summary.capacity
            datastore_summary['name'] = summary.name
            datastore_summary['freeSpace'] = summary.freeSpace
            datastore_summary['maintenanceMode'] = summary.maintenanceMode
            datastore_summary['multipleHostAccess'] = summary.multipleHostAccess
            datastore_summary['type'] = summary.type
            if self.gather_nfs_mount_info or self.gather_vmfs_mount_info:
                if self.gather_nfs_mount_info and summary.type.startswith('NFS'):
                    host_mount_info = self.check_datastore_host(summary.datastore.host[0].key.name, summary.name)
                    datastore_summary['nfs_server'] = host_mount_info.volume.remoteHost
                    datastore_summary['nfs_path'] = host_mount_info.volume.remotePath
                if self.gather_vmfs_mount_info and summary.type == 'VMFS':
                    host_mount_info = self.check_datastore_host(summary.datastore.host[0].key.name, summary.name)
                    datastore_summary['vmfs_blockSize'] = host_mount_info.volume.blockSize
                    datastore_summary['vmfs_version'] = host_mount_info.volume.version
                    datastore_summary['vmfs_uuid'] = host_mount_info.volume.uuid
            if not summary.uncommitted:
                summary.uncommitted = 0
            datastore_summary['uncommitted'] = summary.uncommitted
            datastore_summary['url'] = summary.url
            datastore_summary['provisioned'] = summary.capacity - summary.freeSpace + summary.uncommitted
            datastore_summary['datastore_cluster'] = 'N/A'
            if isinstance(datastore.parent, vim.StoragePod):
                datastore_summary['datastore_cluster'] = datastore.parent.name
            if self.module.params['show_tag']:
                datastore_summary['tags'] = self.vmware_client.get_tags_for_datastore(datastore._moId)
            if self.module.params['name']:
                if datastore_summary['name'] == self.module.params['name']:
                    datastores.extend([datastore_summary])
            else:
                datastores.extend([datastore_summary])
        else:
            temp_ds = self.to_json(datastore, self.properties)
            if self.module.params['show_tag']:
                temp_ds.update({'tags': self.vmware_client.get_tags_for_datastore(datastore._moId)})
            if self.module.params['name']:
                if datastore.name == self.module.params['name']:
                    datastores.extend([temp_ds])
            else:
                datastores.extend([temp_ds])
    return datastores