from libcloud.utils.misc import reverse_dict
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.loadbalancer.base import DEFAULT_ALGORITHM, Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State, Provider
def balancer_list_members(self, balancer):
    members = self._sync_request(command='listLoadBalancerRuleInstances', params={'id': balancer.id}, method='GET')
    members = members['loadbalancerruleinstance']
    return [self._to_member(m, balancer.ex_private_port, balancer) for m in members]