from boto.connection import AWSQueryConnection
from boto.ec2.instanceinfo import InstanceInfo
from boto.ec2.elb.loadbalancer import LoadBalancer, LoadBalancerZones
from boto.ec2.elb.instancestate import InstanceState
from boto.ec2.elb.healthcheck import HealthCheck
from boto.regioninfo import RegionInfo, get_regions, load_regions
from boto.regioninfo import connect
import boto
from boto.compat import six
def apply_security_groups_to_lb(self, name, security_groups):
    """
        Associates one or more security groups with the load balancer.
        The provided security groups will override any currently applied
        security groups.

        :type name: string
        :param name: The name of the Load Balancer

        :type security_groups: List of strings
        :param security_groups: The name of the security group(s) to add.

        :rtype: List of strings
        :return: An updated list of security groups for this Load Balancer.

        """
    params = {'LoadBalancerName': name}
    self.build_list_params(params, security_groups, 'SecurityGroups.member.%d')
    return self.get_list('ApplySecurityGroupsToLoadBalancer', params, None)