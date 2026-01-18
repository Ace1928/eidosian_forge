import boto3
import os
import json
import re
from datetime import datetime
from botocore.exceptions import ClientError
from botocore.exceptions import ProfileNotFound
def send_test_notif(topic_arn, event_type):
    client = get_mturk_client(True)
    client.send_test_event_notification(Notification={'Destination': topic_arn, 'Transport': 'SNS', 'Version': '2006-05-05', 'EventTypes': ['AssignmentAbandoned', 'AssignmentReturned', 'AssignmentSubmitted']}, TestEventType=event_type)