from boto.ec2.elb.listelement import ListElement
from boto.resultset import ResultSet
from boto.ec2.autoscale.launchconfig import LaunchConfiguration
from boto.ec2.autoscale.request import Request
from boto.ec2.autoscale.instance import Instance
from boto.ec2.autoscale.tag import Tag
def get_activities(self, activity_ids=None, max_records=50):
    """
        Get all activies for this group.
        """
    return self.connection.get_all_activities(self, activity_ids, max_records)