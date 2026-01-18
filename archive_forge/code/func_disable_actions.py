from datetime import datetime
from boto.ec2.cloudwatch.listelement import ListElement
from boto.ec2.cloudwatch.dimension import Dimension
from boto.compat import json
from boto.compat import six
def disable_actions(self):
    return self.connection.disable_alarm_actions([self.name])