from octavia_lib.api.drivers import exceptions
def l7policy_delete(self, l7policy):
    """Deletes an L7 policy.

        :param l7policy: The L7 policy to delete.
        :type l7policy: object
        :return: Nothing if the delete request was accepted.
        :raises DriverError: An unexpected error occurred in the driver.
        :raises NotImplementedError: if driver does not support request.
        """
    raise exceptions.NotImplementedError(user_fault_string='This provider does not support deleting l7policies.', operator_fault_string='This provider does not support deleting l7policies.')