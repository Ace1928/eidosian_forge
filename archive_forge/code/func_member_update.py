from octavia_lib.api.drivers import exceptions
def member_update(self, old_member, new_member):
    """Updates a pool member.

        :param old_member: The baseline member object.
        :type old_member: object
        :param new_member: The updated member object.
        :type new_member: object
        :return: Nothing if the create request was accepted.
        :raises DriverError: An unexpected error occurred in the driver.
        :raises NotImplementedError: if driver does not support request.
        :raises UnsupportedOptionError: if driver does not
          support one of the configuration options.
        """
    raise exceptions.NotImplementedError(user_fault_string='This provider does not support updating members.', operator_fault_string='This provider does not support updating members.')