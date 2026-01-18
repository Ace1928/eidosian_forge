from octavia_lib.api.drivers import exceptions
def listener_delete(self, listener):
    """Deletes a listener.

        :param listener: The listener to delete.
        :type listener: object
        :return: Nothing if the delete request was accepted.
        :raises DriverError: An unexpected error occurred in the driver.
        :raises NotImplementedError: if driver does not support request.
        """
    raise exceptions.NotImplementedError(user_fault_string='This provider does not support deleting listeners.', operator_fault_string='This provider does not support deleting listeners.')