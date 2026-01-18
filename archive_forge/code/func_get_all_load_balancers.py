from boto.connection import AWSQueryConnection
from boto.ec2.instanceinfo import InstanceInfo
from boto.ec2.elb.loadbalancer import LoadBalancer, LoadBalancerZones
from boto.ec2.elb.instancestate import InstanceState
from boto.ec2.elb.healthcheck import HealthCheck
from boto.regioninfo import RegionInfo, get_regions, load_regions
from boto.regioninfo import connect
import boto
from boto.compat import six
def get_all_load_balancers(self, load_balancer_names=None, marker=None):
    """
        Retrieve all load balancers associated with your account.

        :type load_balancer_names: list
        :keyword load_balancer_names: An optional list of load balancer names.

        :type marker: string
        :param marker: Use this only when paginating results and only
            in follow-up request after you've received a response
            where the results are truncated.  Set this to the value of
            the Marker element in the response you just received.

        :rtype: :py:class:`boto.resultset.ResultSet`
        :return: A ResultSet containing instances of
            :class:`boto.ec2.elb.loadbalancer.LoadBalancer`
        """
    params = {}
    if load_balancer_names:
        self.build_list_params(params, load_balancer_names, 'LoadBalancerNames.member.%d')
    if marker:
        params['Marker'] = marker
    return self.get_list('DescribeLoadBalancers', params, [('member', LoadBalancer)])