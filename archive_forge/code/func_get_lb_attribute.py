from boto.connection import AWSQueryConnection
from boto.ec2.instanceinfo import InstanceInfo
from boto.ec2.elb.loadbalancer import LoadBalancer, LoadBalancerZones
from boto.ec2.elb.instancestate import InstanceState
from boto.ec2.elb.healthcheck import HealthCheck
from boto.regioninfo import RegionInfo, get_regions, load_regions
from boto.regioninfo import connect
import boto
from boto.compat import six
def get_lb_attribute(self, load_balancer_name, attribute):
    """Gets an attribute of a Load Balancer

        This will make an EC2 call for each method call.

        :type load_balancer_name: string
        :param load_balancer_name: The name of the Load Balancer

        :type attribute: string
        :param attribute: The attribute you wish to see.

          * accessLog - :py:class:`AccessLogAttribute` instance
          * crossZoneLoadBalancing - Boolean
          * connectingSettings - :py:class:`ConnectionSettingAttribute` instance
          * connectionDraining - :py:class:`ConnectionDrainingAttribute`
            instance

        :rtype: Attribute dependent
        :return: The new value for the attribute
        """
    attributes = self.get_all_lb_attributes(load_balancer_name)
    if attribute.lower() == 'accesslog':
        return attributes.access_log
    if attribute.lower() == 'crosszoneloadbalancing':
        return attributes.cross_zone_load_balancing.enabled
    if attribute.lower() == 'connectiondraining':
        return attributes.connection_draining
    if attribute.lower() == 'connectingsettings':
        return attributes.connecting_settings
    return None