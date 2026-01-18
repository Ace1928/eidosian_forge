from libcloud.utils.misc import reverse_dict
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.loadbalancer.base import DEFAULT_ALGORITHM, Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State, Provider
def list_balancers(self):
    balancers = self._sync_request(command='listLoadBalancerRules', method='GET')
    balancers = balancers.get('loadbalancerrule', [])
    return [self._to_balancer(balancer) for balancer in balancers]