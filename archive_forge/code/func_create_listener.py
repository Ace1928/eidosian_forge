from boto.ec2.elb.healthcheck import HealthCheck
from boto.ec2.elb.listener import Listener
from boto.ec2.elb.listelement import ListElement
from boto.ec2.elb.policies import Policies, OtherPolicy
from boto.ec2.elb.securitygroup import SecurityGroup
from boto.ec2.instanceinfo import InstanceInfo
from boto.resultset import ResultSet
from boto.compat import six
def create_listener(self, inPort, outPort=None, proto='tcp'):
    if outPort is None:
        outPort = inPort
    return self.create_listeners([(inPort, outPort, proto)])