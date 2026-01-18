from boto.connection import AWSQueryConnection
from boto.ec2.instanceinfo import InstanceInfo
from boto.ec2.elb.loadbalancer import LoadBalancer, LoadBalancerZones
from boto.ec2.elb.instancestate import InstanceState
from boto.ec2.elb.healthcheck import HealthCheck
from boto.regioninfo import RegionInfo, get_regions, load_regions
from boto.regioninfo import connect
import boto
from boto.compat import six
def delete_lb_policy(self, lb_name, policy_name):
    """
        Deletes a policy from the LoadBalancer. The specified policy must not
        be enabled for any listeners.
        """
    params = {'LoadBalancerName': lb_name, 'PolicyName': policy_name}
    return self.get_status('DeleteLoadBalancerPolicy', params)