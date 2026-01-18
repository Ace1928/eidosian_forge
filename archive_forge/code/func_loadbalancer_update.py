from octavia_lib.api.drivers import exceptions
def loadbalancer_update(self, old_loadbalancer, new_loadbalancer):
    """Updates a load balancer.

        :param old_loadbalancer: The baseline load balancer object.
        :type old_loadbalancer: object
        :param new_loadbalancer: The updated load balancer object.
        :type new_loadbalancer: object
        :return: Nothing if the update request was accepted.
        :raises DriverError: An unexpected error occurred in the driver.
        :raises NotImplementedError: The driver does not support request.
        :raises UnsupportedOptionError: The driver does not
          support one of the configuration options.
        """
    raise exceptions.NotImplementedError(user_fault_string='This provider does not support updating load balancers.', operator_fault_string='This provider does not support updating load balancers.')