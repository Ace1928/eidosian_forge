from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def get_disks(self, container_type, node=None):
    """
        Check for owned disks, unassigned disks or spare disks.
        Return: list of disks or an empty list
        """
    if self.use_rest:
        api = 'storage/disks'
        if container_type == 'owned':
            query = {'home_node.name': node, 'container_type': '!unassigned', 'fields': 'name'}
        if container_type == 'unassigned':
            query = {'container_type': 'unassigned', 'fields': 'name'}
        if container_type == 'spare':
            query = {'home_node.name': node, 'container_type': 'spare', 'fields': 'name'}
        if 'disk_type' in self.parameters:
            query['type'] = self.parameters['disk_type']
        message, error = self.rest_api.get(api, query)
        records, error = rrh.check_for_0_or_more_records(api, message, error)
        if error:
            self.module.fail_json(msg=error)
        return records if records else list()
    else:
        disk_iter = netapp_utils.zapi.NaElement('storage-disk-get-iter')
        disk_storage_info = netapp_utils.zapi.NaElement('storage-disk-info')
        if container_type == 'owned':
            disk_ownership_info = netapp_utils.zapi.NaElement('disk-ownership-info')
            disk_ownership_info.add_new_child('home-node-name', self.parameters['node'])
            disk_storage_info.add_child_elem(disk_ownership_info)
        if container_type == 'unassigned':
            disk_raid_info = netapp_utils.zapi.NaElement('disk-raid-info')
            disk_raid_info.add_new_child('container-type', 'unassigned')
            disk_storage_info.add_child_elem(disk_raid_info)
        disk_query = netapp_utils.zapi.NaElement('query')
        if 'disk_type' in self.parameters and container_type in ('unassigned', 'owned'):
            disk_inventory_info = netapp_utils.zapi.NaElement('disk-inventory-info')
            disk_inventory_info.add_new_child('disk-type', self.parameters['disk_type'])
            disk_query.add_child_elem(disk_inventory_info)
        if container_type == 'spare':
            disk_ownership_info = netapp_utils.zapi.NaElement('disk-ownership-info')
            disk_raid_info = netapp_utils.zapi.NaElement('disk-raid-info')
            disk_ownership_info.add_new_child('owner-node-name', node)
            if 'disk_type' in self.parameters:
                disk_inventory_info = netapp_utils.zapi.NaElement('disk-inventory-info')
                disk_inventory_info.add_new_child('disk-type', self.parameters['disk_type'])
                disk_storage_info.add_child_elem(disk_inventory_info)
            disk_raid_info.add_new_child('container-type', 'spare')
            disk_storage_info.add_child_elem(disk_ownership_info)
            disk_storage_info.add_child_elem(disk_raid_info)
        disk_query.add_child_elem(disk_storage_info)
        disk_iter.add_child_elem(disk_query)
        try:
            result = self.server.invoke_successfully(disk_iter, True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error getting disk information: %s' % to_native(error), exception=traceback.format_exc())
        disks = []
        if result.get_child_by_name('attributes-list'):
            attributes_list = result.get_child_by_name('attributes-list')
            storage_disk_info_attributes = attributes_list.get_children()
            for disk in storage_disk_info_attributes:
                disk_inventory_info = disk.get_child_by_name('disk-inventory-info')
                disk_name = disk_inventory_info.get_child_content('disk-cluster-name')
                disks.append(disk_name)
        return disks