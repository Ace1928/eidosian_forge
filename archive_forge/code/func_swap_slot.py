from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def swap_slot(self):
    """
        Swap slot
        :return: deserialized response
        """
    self.log('Swap slot')
    try:
        if self.swap['action'] == 'swap':
            if self.swap['target_slot'] is None:
                slot_swap_entity = CsmSlotEntity(target_slot=self.name, preserve_vnet=self.swap['preserve_vnet'])
                response = self.web_client.web_apps.begin_swap_slot_with_production(resource_group_name=self.resource_group, name=self.webapp_name, slot_swap_entity=slot_swap_entity)
            else:
                slot_swap_entity = CsmSlotEntity(target_slot=self.swap['target_slot'], preserve_vnet=self.swap['preserve_vnet'])
                response = self.web_client.web_apps.begin_swap_slot(resource_group_name=self.resource_group, name=self.webapp_name, slot=self.name, slot_swap_entity=slot_swap_entity)
        elif self.swap['action'] == 'preview':
            if self.swap['target_slot'] is None:
                slot_swap_entity = CsmSlotEntity(target_slot=self.name, preserve_vnet=self.swap['preserve_vnet'])
                response = self.web_client.web_apps.apply_slot_config_to_production(resource_group_name=self.resource_group, name=self.webapp_name, slot_swap_entity=slot_swap_entity)
            else:
                slot_swap_entity = CsmSlotEntity(target_slot=self.swap['target_slot'], preserve_vnet=self.swap['preserve_vnet'])
                response = self.web_client.web_apps.apply_slot_configuration_slot(resource_group_name=self.resource_group, name=self.webapp_name, slot=self.name, slot_swap_entity=slot_swap_entity)
        elif self.swap['action'] == 'reset':
            if self.swap['target_slot'] is None:
                response = self.web_client.web_apps.reset_production_slot_config(resource_group_name=self.resource_group, name=self.webapp_name)
            else:
                response = self.web_client.web_apps.reset_slot_configuration_slot(resource_group_name=self.resource_group, name=self.webapp_name, slot=self.swap['target_slot'])
            response = self.web_client.web_apps.reset_slot_configuration_slot(resource_group_name=self.resource_group, name=self.webapp_name, slot=self.name)
        self.log('Response : {0}'.format(response))
        return response
    except Exception as ex:
        self.fail('Failed to swap web app slot {0} in resource group {1}: {2}'.format(self.name, self.resource_group, str(ex)))