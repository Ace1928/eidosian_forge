from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def create_or_update_vmextension(self):
    """
        Method calling the Azure SDK to create or update the VM extension.
        :return: void
        """
    self.log('Creating VM extension {0}'.format(self.name))
    try:
        params = self.compute_models.VirtualMachineExtension(location=self.location, publisher=self.publisher, type_properties_type=self.virtual_machine_extension_type, type_handler_version=self.type_handler_version, auto_upgrade_minor_version=self.auto_upgrade_minor_version, settings=self.settings, protected_settings=self.protected_settings, force_update_tag=self.force_update_tag)
        poller = self.compute_client.virtual_machine_extensions.begin_create_or_update(self.resource_group, self.virtual_machine_name, self.name, params)
        response = self.get_poller_result(poller)
        return vmextension_to_dict(response)
    except Exception as e:
        self.log('Error attempting to create the VM extension.')
        self.fail('Error creating the VM extension: {0}'.format(str(e)))