from octavia_lib.api.drivers import exceptions
def health_monitor_create(self, healthmonitor):
    """Creates a new health monitor.

        :param healthmonitor: The health monitor object.
        :type healthmonitor: object
        :return: Nothing if the create request was accepted.
        :raises DriverError: An unexpected error occurred in the driver.
        :raises NotImplementedError: if driver does not support request.
        :raises UnsupportedOptionError: if driver does not
          support one of the configuration options.
        """
    raise exceptions.NotImplementedError(user_fault_string='This provider does not support creating health monitors.', operator_fault_string='This provider does not support creating health monitors.')