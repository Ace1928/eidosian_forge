from __future__ import absolute_import, division, print_function
def get_by_role_name(self):
    """
        Get Role Definition in scope by role name.

        :return: deserialized role definition state dictionary
        """
    self.log('Get Role Definition by name {0}'.format(self.role_name))
    response = []
    try:
        response = self.list()
        if len(response) > 0:
            roles = []
            for r in response:
                if r['role_name'] == self.role_name:
                    roles.append(r)
            if len(roles) == 1:
                self.log('Role Definition : {0} found'.format(self.role_name))
                return roles
            if len(roles) > 1:
                self.fail('Found multiple Role Definitions with name: {0}'.format(self.role_name))
    except Exception as ex:
        self.log("Didn't find Role Definition by name {0}".format(self.role_name))
    return []