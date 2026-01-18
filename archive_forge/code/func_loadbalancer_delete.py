from octavia_lib.api.drivers import exceptions
def loadbalancer_delete(self, loadbalancer, cascade=False):
    """Deletes a load balancer.

        :param loadbalancer: The load balancer to delete.
        :type loadbalancer: object
        :param cascade: If True, deletes all child objects (listeners,
          pools, etc.) in addition to the load balancer.
        :type cascade: bool
        :return: Nothing if the delete request was accepted.
        :raises DriverError: An unexpected error occurred in the driver.
        :raises NotImplementedError: if driver does not support request.
        """
    raise exceptions.NotImplementedError(user_fault_string='This provider does not support deleting load balancers.', operator_fault_string='This provider does not support deleting load balancers.')