from __future__ import absolute_import, division, print_function
import uuid
from ansible.module_utils._text import to_native
def delete_roledefinition(self, role_definition_id):
    """
        Deletes specified role definition.

        :return: True
        """
    self.log('Deleting the role definition {0}'.format(self.name))
    scope = self.build_scope()
    try:
        response = self._client.role_definitions.delete(scope=scope, role_definition_id=role_definition_id)
        if isinstance(response, LROPoller):
            response = self.get_poller_result(response)
    except Exception as e:
        self.log('Error attempting to delete the role definition.')
        self.fail('Error deleting the role definition: {0}'.format(str(e)))
    return True