from datetime import datetime
from boto.ec2.cloudwatch.listelement import ListElement
from boto.ec2.cloudwatch.dimension import Dimension
from boto.compat import json
from boto.compat import six
def describe_history(self, start_date=None, end_date=None, max_records=None, history_item_type=None, next_token=None):
    return self.connection.describe_alarm_history(self.name, start_date, end_date, max_records, history_item_type, next_token)