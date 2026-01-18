from octavia_lib.api.drivers import exceptions
def member_create(self, member):
    """Creates a new member for a pool.

        :param member: The member object.
        :type member: object
        :return: Nothing if the create request was accepted.
        :raises DriverError: An unexpected error occurred in the driver.
        :raises NotImplementedError: if driver does not support request.
        :raises UnsupportedOptionError: if driver does not
          support one of the configuration options.
        """
    raise exceptions.NotImplementedError(user_fault_string='This provider does not support creating members.', operator_fault_string='This provider does not support creating members.')