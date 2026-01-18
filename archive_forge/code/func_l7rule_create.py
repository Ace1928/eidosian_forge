from octavia_lib.api.drivers import exceptions
def l7rule_create(self, l7rule):
    """Creates a new L7 rule.

        :param l7rule: The L7 rule object.
        :type l7rule: object
        :return: Nothing if the create request was accepted.
        :raises DriverError: An unexpected error occurred in the driver.
        :raises NotImplementedError: if driver does not support request.
        :raises UnsupportedOptionError: if driver does not
          support one of the configuration options.
        """
    raise exceptions.NotImplementedError(user_fault_string='This provider does not support creating l7rules.', operator_fault_string='This provider does not support creating l7rules.')