from boto.ec2.elb.healthcheck import HealthCheck
from boto.ec2.elb.listener import Listener
from boto.ec2.elb.listelement import ListElement
from boto.ec2.elb.policies import Policies, OtherPolicy
from boto.ec2.elb.securitygroup import SecurityGroup
from boto.ec2.instanceinfo import InstanceInfo
from boto.resultset import ResultSet
from boto.compat import six
def detach_subnets(self, subnets):
    """
        Detaches load balancer from one or more subnets.

        :type subnets: string or List of strings
        :param subnets: The name of the subnet(s) to detach.

        """
    if isinstance(subnets, six.string_types):
        subnets = [subnets]
    new_subnets = self.connection.detach_lb_from_subnets(self.name, subnets)
    self.subnets = new_subnets