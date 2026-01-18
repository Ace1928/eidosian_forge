from octavia_lib.api.drivers import exceptions
def listener_update(self, old_listener, new_listener):
    """Updates a listener.

        :param old_listener: The baseline listener object.
        :type old_listener: object
        :param new_listener: The updated listener object.
        :type new_listener: object
        :return: Nothing if the update request was accepted.
        :raises DriverError: An unexpected error occurred in the driver.
        :raises NotImplementedError: if driver does not support request.
        :raises UnsupportedOptionError: if driver does not
          support one of the configuration options.
        """
    raise exceptions.NotImplementedError(user_fault_string='This provider does not support updating listeners.', operator_fault_string='This provider does not support updating listeners.')