from boto.connection import AWSQueryConnection
from boto.ec2.instanceinfo import InstanceInfo
from boto.ec2.elb.loadbalancer import LoadBalancer, LoadBalancerZones
from boto.ec2.elb.instancestate import InstanceState
from boto.ec2.elb.healthcheck import HealthCheck
from boto.regioninfo import RegionInfo, get_regions, load_regions
from boto.regioninfo import connect
import boto
from boto.compat import six
def delete_load_balancer_listeners(self, name, ports):
    """
        Deletes a load balancer listener (or group of listeners)

        :type name: string
        :param name: The name of the load balancer to create the listeners for

        :type ports: List int
        :param ports: Each int represents the port on the ELB to be removed

        :return: The status of the request
        """
    params = {'LoadBalancerName': name}
    for index, port in enumerate(ports):
        params['LoadBalancerPorts.member.%d' % (index + 1)] = port
    return self.get_status('DeleteLoadBalancerListeners', params)