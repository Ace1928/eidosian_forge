from __future__ import absolute_import, division, print_function
import uuid
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
def create_update_resource(self):
    try:
        response = self.mgmt_client.registration_definitions.begin_create_or_update(registration_definition_id=self.registration_definition_id, scope=self.scope, request_body=self.body)
        if isinstance(response, LROPoller):
            response = self.get_poller_result(response)
    except Exception as exc:
        self.log('Error attempting to create the RegistrationDefinition instance.')
        self.fail('Error creating the RegistrationDefinition instance: {0}'.format(str(exc)))
    return response.as_dict()