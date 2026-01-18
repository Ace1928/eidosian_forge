from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
def gather_first_class_disk_info(self):
    self.datastore_obj = self.find_datastore_by_name(datastore_name=self.datastore_name, datacenter_name=self.datacenter_name)
    if not self.datastore_obj:
        self.module.fail_json(msg='Failed to find datastore %s.' % self.datastore_name)
    if self.disk_name:
        self.disk_obj = self.find_first_class_disk_by_name(self.disk_name, self.datastore_obj)
        if not self.disk_obj:
            self.module.fail_json(msg='Failed to find disk %s.' % self.disk_name)
        self.disks = [self.disk_obj]
    else:
        self.disks = self.find_first_class_disks(self.datastore_obj)
        if not self.disks:
            return []
    disk_infos = list()
    for disk in self.disks:
        disk_info = dict(name=disk.config.name, id=disk.config.id.id, datastore_name=disk.config.backing.datastore.name, size_mb=disk.config.capacityInMB, consumption_type=disk.config.consumptionType, descriptor_version=disk.config.descriptorVersion, consumer_ids=list((id.id for id in disk.config.consumerId)))
        disk_infos.append(disk_info)
    return disk_infos