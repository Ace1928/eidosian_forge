from boto.connection import AWSQueryConnection
from boto.ec2.instanceinfo import InstanceInfo
from boto.ec2.elb.loadbalancer import LoadBalancer, LoadBalancerZones
from boto.ec2.elb.instancestate import InstanceState
from boto.ec2.elb.healthcheck import HealthCheck
from boto.regioninfo import RegionInfo, get_regions, load_regions
from boto.regioninfo import connect
import boto
from boto.compat import six
def regions():
    """
    Get all available regions for the ELB service.

    :rtype: list
    :return: A list of :class:`boto.RegionInfo` instances
    """
    return get_regions('elasticloadbalancing', connection_cls=ELBConnection)