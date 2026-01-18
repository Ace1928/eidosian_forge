from octavia_lib.api.drivers import exceptions
def loadbalancer_create(self, loadbalancer):
    """Creates a new load balancer.

        :param loadbalancer: The load balancer object.
        :type loadbalancer: object
        :return: Nothing if the create request was accepted.
        :raises DriverError: An unexpected error occurred in the driver.
        :raises NotImplementedError: The driver does not support create.
        :raises UnsupportedOptionError: The driver does not
          support one of the configuration options.
        """
    raise exceptions.NotImplementedError(user_fault_string='This provider does not support creating load balancers.', operator_fault_string='This provider does not support creating load balancers. What?')