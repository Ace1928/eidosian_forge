from __future__ import absolute_import, division, print_function
def roleassignment_to_dict(self, assignment):
    return dict(assignee_object_id=assignment.principal_id, id=assignment.id, name=assignment.name, principal_id=assignment.principal_id, principal_type=assignment.principal_type, role_definition_id=assignment.role_definition_id, scope=assignment.scope, type=assignment.type)