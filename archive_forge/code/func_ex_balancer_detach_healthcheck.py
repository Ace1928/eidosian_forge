from libcloud.loadbalancer.base import Driver, Member, Algorithm, LoadBalancer
from libcloud.compute.drivers.gce import GCEConnection, GCENodeDriver
def ex_balancer_detach_healthcheck(self, balancer, healthcheck):
    """
        Detach healtcheck from balancer

        :param balancer: LoadBalancer which should be used
        :type  balancer: :class:`LoadBalancer`

        :param healthcheck: Healthcheck to remove
        :type  healthcheck: :class:`GCEHealthCheck`

        :return: True if successful
        :rtype: ``bool``
        """
    return balancer.extra['targetpool'].remove_healthcheck(healthcheck)