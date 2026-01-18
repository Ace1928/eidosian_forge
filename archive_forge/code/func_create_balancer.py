from libcloud.utils.misc import reverse_dict
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.loadbalancer.base import DEFAULT_ALGORITHM, Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State, Provider
def create_balancer(self, name, members, protocol='http', port=80, algorithm=DEFAULT_ALGORITHM, location=None, private_port=None, network_id=None, vpc_id=None):
    """
        @inherits: :class:`Driver.create_balancer`

        :param location: Location
        :type  location: :class:`NodeLocation`

        :param private_port: Private port
        :type  private_port: ``int``

        :param network_id: The guest network this rule will be created for.
        :type  network_id: ``str``
        """
    args = {}
    ip_args = {}
    if location is None:
        locations = self._sync_request(command='listZones', method='GET')
        location = locations['zone'][0]['id']
    else:
        location = location.id
    if private_port is None:
        private_port = port
    if network_id is not None:
        args['networkid'] = network_id
        ip_args['networkid'] = network_id
    if vpc_id is not None:
        ip_args['vpcid'] = vpc_id
    ip_args.update({'zoneid': location, 'networkid': network_id, 'vpc_id': vpc_id})
    result = self._async_request(command='associateIpAddress', params=ip_args, method='GET')
    public_ip = result['ipaddress']
    args.update({'algorithm': self._ALGORITHM_TO_VALUE_MAP[algorithm], 'name': name, 'privateport': private_port, 'publicport': port, 'publicipid': public_ip['id']})
    result = self._sync_request(command='createLoadBalancerRule', params=args, method='GET')
    listbalancers = self._sync_request(command='listLoadBalancerRules', params=args, method='GET')
    listbalancers = [rule for rule in listbalancers['loadbalancerrule'] if rule['id'] == result['id']]
    if len(listbalancers) != 1:
        return None
    balancer = self._to_balancer(listbalancers[0])
    for member in members:
        balancer.attach_member(member)
    return balancer