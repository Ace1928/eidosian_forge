from boto.connection import AWSQueryConnection
from boto.ec2.instanceinfo import InstanceInfo
from boto.ec2.elb.loadbalancer import LoadBalancer, LoadBalancerZones
from boto.ec2.elb.instancestate import InstanceState
from boto.ec2.elb.healthcheck import HealthCheck
from boto.regioninfo import RegionInfo, get_regions, load_regions
from boto.regioninfo import connect
import boto
from boto.compat import six
def set_lb_listener_SSL_certificate(self, lb_name, lb_port, ssl_certificate_id):
    """
        Sets the certificate that terminates the specified listener's SSL
        connections. The specified certificate replaces any prior certificate
        that was used on the same LoadBalancer and port.
        """
    params = {'LoadBalancerName': lb_name, 'LoadBalancerPort': lb_port, 'SSLCertificateId': ssl_certificate_id}
    return self.get_status('SetLoadBalancerListenerSSLCertificate', params)