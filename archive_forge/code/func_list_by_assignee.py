from __future__ import absolute_import, division, print_function
def list_by_assignee(self):
    """
        Gets the role assignments by assignee.

        :return: deserialized role assignment dictionary
        """
    self.log('Gets role assignment {0} by name'.format(self.name))
    filter = "principalId eq '{0}'".format(self.assignee)
    return self.list_assignments(filter=filter)