from __future__ import absolute_import, division, print_function
def list_by_scope(self):
    """
        Lists the role assignments by specific scope.

        :return: deserialized role assignment dictionary
        """
    self.log('Lists role assignment by scope {0}'.format(self.scope))
    results = []
    try:
        response = list(self.authorization_client.role_assignments.list_for_scope(scope=self.scope, filter='atScope()'))
        response = [self.roleassignment_to_dict(role_assignment) for role_assignment in response]
        if self.assignee:
            response = [role_assignment for role_assignment in response if role_assignment.get('principal_id').lower() == self.assignee.lower()]
        if self.strict_scope_match:
            response = [role_assignment for role_assignment in response if role_assignment.get('scope').lower() == self.scope.lower()]
        if self.role_definition_id:
            response = [role_assignment for role_assignment in response if role_assignment.get('role_definition_id').split('/')[-1].lower() == self.role_definition_id.split('/')[-1].lower()]
        results = response
    except Exception as ex:
        self.log("Didn't find role assignments at scope {0}".format(self.scope))
    return results