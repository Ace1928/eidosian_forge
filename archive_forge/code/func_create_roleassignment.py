from __future__ import absolute_import, division, print_function
def create_roleassignment(self):
    """
        Creates role assignment.

        :return: deserialized role assignment
        """
    self.log('Creating role assignment {0}'.format(self.name))
    response = None
    try:
        parameters = self.authorization_models.RoleAssignmentCreateParameters(role_definition_id=self.role_definition_id, principal_id=self.assignee_object_id)
        if self.id:
            response = self.authorization_client.role_assignments.create_by_id(role_id=self.id, parameters=parameters)
        elif self.scope:
            if not self.name:
                self.name = str(uuid.uuid4())
            response = self.authorization_client.role_assignments.create(scope=self.scope, role_assignment_name=self.name, parameters=parameters)
    except Exception as exc:
        self.log('Error attempting to create role assignment.')
        self.fail('Error creating role assignment: {0}'.format(str(exc)))
    return self.roleassignment_to_dict(response)