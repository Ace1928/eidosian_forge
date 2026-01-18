from __future__ import absolute_import, division, print_function
import uuid
from ansible.module_utils._text import to_native
def create_update_roledefinition(self):
    """
        Creates or updates role definition.

        :return: deserialized role definition
        """
    self.log('Creating / Updating role definition {0}'.format(self.name))
    try:
        permissions = None
        if self.permissions:
            permissions = [AuthorizationManagementClient.models('2018-01-01-preview').Permission(actions=p.get('actions', None), not_actions=p.get('not_actions', None), data_actions=p.get('data_actions', None), not_data_actions=p.get('not_data_actions', None)) for p in self.permissions]
        role_definition = AuthorizationManagementClient.models('2018-01-01-preview').RoleDefinition(role_name=self.name, description=self.description, permissions=permissions, assignable_scopes=self.assignable_scopes, role_type='CustomRole')
        if self.role:
            role_definition.name = self.role['name']
        response = self._client.role_definitions.create_or_update(role_definition_id=self.role['name'] if self.role else str(uuid.uuid4()), scope=self.scope, role_definition=role_definition)
        if isinstance(response, LROPoller):
            response = self.get_poller_result(response)
    except Exception as exc:
        self.log('Error attempting to create role definition.')
        self.fail('Error creating role definition: {0}'.format(str(exc)))
    return roledefinition_to_dict(response)