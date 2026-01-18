from boto.connection import AWSQueryConnection
from boto.ec2.instanceinfo import InstanceInfo
from boto.ec2.elb.loadbalancer import LoadBalancer, LoadBalancerZones
from boto.ec2.elb.instancestate import InstanceState
from boto.ec2.elb.healthcheck import HealthCheck
from boto.regioninfo import RegionInfo, get_regions, load_regions
from boto.regioninfo import connect
import boto
from boto.compat import six
def set_lb_policies_of_listener(self, lb_name, lb_port, policies):
    """
        Associates, updates, or disables a policy with a listener on the load
        balancer. Currently only zero (0) or one (1) policy can be associated
        with a listener.
        """
    params = {'LoadBalancerName': lb_name, 'LoadBalancerPort': lb_port}
    if len(policies):
        self.build_list_params(params, policies, 'PolicyNames.member.%d')
    else:
        params['PolicyNames'] = ''
    return self.get_status('SetLoadBalancerPoliciesOfListener', params)