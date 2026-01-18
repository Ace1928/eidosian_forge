from boto.ec2.elb.healthcheck import HealthCheck
from boto.ec2.elb.listener import Listener
from boto.ec2.elb.listelement import ListElement
from boto.ec2.elb.policies import Policies, OtherPolicy
from boto.ec2.elb.securitygroup import SecurityGroup
from boto.ec2.instanceinfo import InstanceInfo
from boto.resultset import ResultSet
from boto.compat import six
def set_listener_SSL_certificate(self, lb_port, ssl_certificate_id):
    return self.connection.set_lb_listener_SSL_certificate(self.name, lb_port, ssl_certificate_id)