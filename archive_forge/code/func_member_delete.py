from octavia_lib.api.drivers import exceptions
def member_delete(self, member):
    """Deletes a pool member.

        :param member: The member to delete.
        :type member: object
        :return: Nothing if the create request was accepted.
        :raises DriverError: An unexpected error occurred in the driver.
        :raises NotImplementedError: if driver does not support request.
        """
    raise exceptions.NotImplementedError(user_fault_string='This provider does not support deleting members.', operator_fault_string='This provider does not support deleting members.')