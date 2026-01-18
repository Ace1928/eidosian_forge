from octavia_lib.api.drivers import exceptions
def listener_create(self, listener):
    """Creates a new listener.

        :param listener: The listener object.
        :type listener: object
        :return: Nothing if the create request was accepted.
        :raises DriverError: An unexpected error occurred in the driver.
        :raises NotImplementedError: if driver does not support request.
        :raises UnsupportedOptionError: if driver does not
          support one of the configuration options.
        """
    raise exceptions.NotImplementedError(user_fault_string='This provider does not support creating listeners.', operator_fault_string='This provider does not support creating listeners.')