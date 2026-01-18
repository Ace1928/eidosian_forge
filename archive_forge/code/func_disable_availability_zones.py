from boto.connection import AWSQueryConnection
from boto.ec2.instanceinfo import InstanceInfo
from boto.ec2.elb.loadbalancer import LoadBalancer, LoadBalancerZones
from boto.ec2.elb.instancestate import InstanceState
from boto.ec2.elb.healthcheck import HealthCheck
from boto.regioninfo import RegionInfo, get_regions, load_regions
from boto.regioninfo import connect
import boto
from boto.compat import six
def disable_availability_zones(self, load_balancer_name, zones_to_remove):
    """
        Remove availability zones from an existing Load Balancer.
        All zones must be in the same region as the Load Balancer.
        Removing zones that are not registered with the Load Balancer
        has no effect.
        You cannot remove all zones from an Load Balancer.

        :type load_balancer_name: string
        :param load_balancer_name: The name of the Load Balancer

        :type zones: List of strings
        :param zones: The name of the zone(s) to remove.

        :rtype: List of strings
        :return: An updated list of zones for this Load Balancer.

        """
    params = {'LoadBalancerName': load_balancer_name}
    self.build_list_params(params, zones_to_remove, 'AvailabilityZones.member.%d')
    obj = self.get_object('DisableAvailabilityZonesForLoadBalancer', params, LoadBalancerZones)
    return obj.zones