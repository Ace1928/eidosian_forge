from libcloud.loadbalancer.base import Driver, Member, Algorithm, LoadBalancer
from libcloud.compute.drivers.gce import GCEConnection, GCENodeDriver
def ex_balancer_attach_healthcheck(self, balancer, healthcheck):
    """
        Attach a healthcheck to balancer

        :param balancer: LoadBalancer which should be used
        :type  balancer: :class:`LoadBalancer`

        :param healthcheck: Healthcheck to add
        :type  healthcheck: :class:`GCEHealthCheck`

        :return: True if successful
        :rtype:  ``bool``
        """
    return balancer.extra['targetpool'].add_healthcheck(healthcheck)