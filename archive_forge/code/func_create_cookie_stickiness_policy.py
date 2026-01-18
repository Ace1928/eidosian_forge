from boto.ec2.elb.healthcheck import HealthCheck
from boto.ec2.elb.listener import Listener
from boto.ec2.elb.listelement import ListElement
from boto.ec2.elb.policies import Policies, OtherPolicy
from boto.ec2.elb.securitygroup import SecurityGroup
from boto.ec2.instanceinfo import InstanceInfo
from boto.resultset import ResultSet
from boto.compat import six
def create_cookie_stickiness_policy(self, cookie_expiration_period, policy_name):
    return self.connection.create_lb_cookie_stickiness_policy(cookie_expiration_period, self.name, policy_name)