from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def create_update_containerregistry(self, to_do):
    """
        Creates or updates a container registry.

        :return: deserialized container registry instance state dictionary
        """
    self.log('Creating / Updating the container registry instance {0}'.format(self.name))
    try:
        if to_do != Actions.NoAction:
            if to_do == Actions.Create:
                name_status = self.containerregistry_client.registries.check_name_availability(registry_name_check_request=RegistryNameCheckRequest(name=self.name))
                if name_status.name_available:
                    poller = self.containerregistry_client.registries.begin_create(resource_group_name=self.resource_group, registry_name=self.name, registry=Registry(location=self.location, sku=Sku(name=self.sku), tags=self.tags, admin_user_enabled=self.admin_user_enabled))
                else:
                    raise Exception('Invalid registry name. reason: ' + name_status.reason + ' message: ' + name_status.message)
            else:
                registry = self.containerregistry_client.registries.get(resource_group_name=self.resource_group, registry_name=self.name)
                if registry is not None:
                    poller = self.containerregistry_client.registries.begin_update(resource_group_name=self.resource_group, registry_name=self.name, registry_update_parameters=RegistryUpdateParameters(sku=Sku(name=self.sku), tags=self.tags, admin_user_enabled=self.admin_user_enabled))
                else:
                    raise Exception("Update registry failed as registry '" + self.name + "' doesn't exist.")
            response = self.get_poller_result(poller)
            if self.admin_user_enabled:
                credentials = self.containerregistry_client.registries.list_credentials(resource_group_name=self.resource_group, registry_name=self.name)
            else:
                self.log('Cannot perform credential operations as admin user is disabled')
                credentials = None
        else:
            response = None
            credentials = None
    except Exception as exc:
        self.log('Error attempting to create / update the container registry instance.')
        self.fail('Error creating / updating the container registry instance: {0}'.format(str(exc)))
    return create_containerregistry_dict(response, credentials)