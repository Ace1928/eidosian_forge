from boto.connection import AWSQueryConnection
from boto.ec2.instanceinfo import InstanceInfo
from boto.ec2.elb.loadbalancer import LoadBalancer, LoadBalancerZones
from boto.ec2.elb.instancestate import InstanceState
from boto.ec2.elb.healthcheck import HealthCheck
from boto.regioninfo import RegionInfo, get_regions, load_regions
from boto.regioninfo import connect
import boto
from boto.compat import six
def modify_lb_attribute(self, load_balancer_name, attribute, value):
    """Changes an attribute of a Load Balancer

        :type load_balancer_name: string
        :param load_balancer_name: The name of the Load Balancer

        :type attribute: string
        :param attribute: The attribute you wish to change.

        * crossZoneLoadBalancing - Boolean (true)
        * connectingSettings - :py:class:`ConnectionSettingAttribute` instance
        * accessLog - :py:class:`AccessLogAttribute` instance
        * connectionDraining - :py:class:`ConnectionDrainingAttribute` instance

        :type value: string
        :param value: The new value for the attribute

        :rtype: bool
        :return: Whether the operation succeeded or not
        """
    bool_reqs = ('crosszoneloadbalancing',)
    if attribute.lower() in bool_reqs:
        if isinstance(value, bool):
            if value:
                value = 'true'
            else:
                value = 'false'
    params = {'LoadBalancerName': load_balancer_name}
    if attribute.lower() == 'crosszoneloadbalancing':
        params['LoadBalancerAttributes.CrossZoneLoadBalancing.Enabled'] = value
    elif attribute.lower() == 'accesslog':
        params['LoadBalancerAttributes.AccessLog.Enabled'] = value.enabled and 'true' or 'false'
        params['LoadBalancerAttributes.AccessLog.S3BucketName'] = value.s3_bucket_name
        params['LoadBalancerAttributes.AccessLog.S3BucketPrefix'] = value.s3_bucket_prefix
        params['LoadBalancerAttributes.AccessLog.EmitInterval'] = value.emit_interval
    elif attribute.lower() == 'connectiondraining':
        params['LoadBalancerAttributes.ConnectionDraining.Enabled'] = value.enabled and 'true' or 'false'
        params['LoadBalancerAttributes.ConnectionDraining.Timeout'] = value.timeout
    elif attribute.lower() == 'connectingsettings':
        params['LoadBalancerAttributes.ConnectionSettings.IdleTimeout'] = value.idle_timeout
    else:
        raise ValueError('InvalidAttribute', attribute)
    return self.get_status('ModifyLoadBalancerAttributes', params, verb='GET')