from boto.ec2.elb.healthcheck import HealthCheck
from boto.ec2.elb.listener import Listener
from boto.ec2.elb.listelement import ListElement
from boto.ec2.elb.policies import Policies, OtherPolicy
from boto.ec2.elb.securitygroup import SecurityGroup
from boto.ec2.instanceinfo import InstanceInfo
from boto.resultset import ResultSet
from boto.compat import six
def enable_zones(self, zones):
    """
        Enable availability zones to this Access Point.
        All zones must be in the same region as the Access Point.

        :type zones: string or List of strings
        :param zones: The name of the zone(s) to add.

        """
    if isinstance(zones, six.string_types):
        zones = [zones]
    new_zones = self.connection.enable_availability_zones(self.name, zones)
    self.availability_zones = new_zones