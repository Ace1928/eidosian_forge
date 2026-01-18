from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi
from ansible.module_utils._text import to_native
def deploy_vm_from_template(self, power_on=False):
    self._datacenter_id = self.get_datacenter_by_name(self.datacenter)
    if not self._datacenter_id:
        self._fail(msg='Failed to find the datacenter %s' % self.datacenter)
    if self.datastore:
        self._datastore_id = self.get_datastore_by_name(self.datacenter, self.datastore)
        if not self._datastore_id:
            self._fail(msg='Failed to find the datastore %s' % self.datastore)
    if self.datastore_cluster and (not self._datastore_id):
        dsc = self._pyv.find_datastore_cluster_by_name(self.datastore_cluster)
        if dsc:
            self.datastore = self._pyv.get_recommended_datastore(dsc)
            self._datastore_id = self.get_datastore_by_name(self.datacenter, self.datastore)
        else:
            self._fail(msg='Failed to find the datastore cluster %s' % self.datastore_cluster)
    if not self._datastore_id:
        self._fail(msg='Failed to find the datastore using either datastore or datastore cluster')
    if self.library:
        self._library_item_id = self.get_library_item_from_content_library_name(self.template, self.library)
        if not self._library_item_id:
            self._fail(msg='Failed to find the library Item %s in content library %s' % (self.template, self.library))
    else:
        self._library_item_id = self.get_library_item_by_name(self.template)
        if not self._library_item_id:
            self._fail(msg='Failed to find the library Item %s' % self.template)
    folder_obj = self._pyv.find_folder_by_fqpn(self.folder, self.datacenter, folder_type='vm')
    if folder_obj:
        self._folder_id = folder_obj._moId
    if not self._folder_id:
        self._fail(msg='Failed to find the folder %s' % self.folder)
    if self.host:
        self._host_id = self.get_host_by_name(self.datacenter, self.host)
        if not self._host_id:
            self._fail(msg='Failed to find the Host %s' % self.host)
    if self.cluster:
        self._cluster_id = self.get_cluster_by_name(self.datacenter, self.cluster)
        if not self._cluster_id:
            self._fail(msg='Failed to find the Cluster %s' % self.cluster)
        cluster_obj = self.api_client.vcenter.Cluster.get(self._cluster_id)
        self._resourcepool_id = cluster_obj.resource_pool
    if self.resourcepool:
        self._resourcepool_id = self.get_resource_pool_by_name(self.datacenter, self.resourcepool, self.cluster, self.host)
        if not self._resourcepool_id:
            self._fail(msg='Failed to find the resource_pool %s' % self.resourcepool)
    self.placement_spec = LibraryItems.DeployPlacementSpec(folder=self._folder_id)
    if self._host_id:
        self.placement_spec.host = self._host_id
    if self._resourcepool_id:
        self.placement_spec.resource_pool = self._resourcepool_id
    if self._cluster_id:
        self.placement_spec.cluster = self._cluster_id
    self.vm_home_storage_spec = LibraryItems.DeploySpecVmHomeStorage(datastore=to_native(self._datastore_id))
    self.disk_storage_spec = LibraryItems.DeploySpecDiskStorage(datastore=to_native(self._datastore_id))
    self.deploy_spec = LibraryItems.DeploySpec(name=self.vm_name, placement=self.placement_spec, vm_home_storage=self.vm_home_storage_spec, disk_storage=self.disk_storage_spec, powered_on=power_on)
    vm_id = ''
    try:
        vm_id = self._template_service.deploy(self._library_item_id, self.deploy_spec)
    except Error as error:
        self._fail(msg='%s' % self.get_error_message(error))
    except Exception as err:
        self._fail(msg='%s' % to_native(err))
    if not vm_id:
        self.result['vm_deploy_info'] = dict(msg='Virtual Machine deployment failed', vm_id='')
        self._fail(msg='Virtual Machine deployment failed')
    self.result['changed'] = True
    self.result['vm_deploy_info'] = dict(msg="Deployed Virtual Machine '%s'." % self.vm_name, vm_id=vm_id)
    self._exit()