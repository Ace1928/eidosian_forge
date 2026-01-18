from libcloud.utils.misc import reverse_dict
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.loadbalancer.base import DEFAULT_ALGORITHM, Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State, Provider
def balancer_attach_member(self, balancer, member):
    member.port = balancer.ex_private_port
    self._async_request(command='assignToLoadBalancerRule', params={'id': balancer.id, 'virtualmachineids': member.id}, method='GET')
    return True