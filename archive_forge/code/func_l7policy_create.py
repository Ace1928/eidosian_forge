from octavia_lib.api.drivers import exceptions
def l7policy_create(self, l7policy):
    """Creates a new L7 policy.

        :param l7policy: The L7 policy object.
        :type l7policy: object
        :return: Nothing if the create request was accepted.
        :raises DriverError: An unexpected error occurred in the driver.
        :raises NotImplementedError: if driver does not support request.
        :raises UnsupportedOptionError: if driver does not
          support one of the configuration options.
        """
    raise exceptions.NotImplementedError(user_fault_string='This provider does not support creating l7policies.', operator_fault_string='This provider does not support creating l7policies.')