import boto3
import os
import json
import re
from datetime import datetime
from botocore.exceptions import ClientError
from botocore.exceptions import ProfileNotFound
def setup_sns_topic(task_name, server_url, task_group_id):
    client = boto3.client('sns', region_name='us-east-1')
    pattern = re.compile('[^a-zA-Z0-9_-]+')
    filtered_task_name = pattern.sub('', task_name)
    response = client.create_topic(Name=filtered_task_name)
    arn = response['TopicArn']
    topic_sub_url = '{}/sns_posts?task_group_id={}'.format(server_url, task_group_id)
    client.subscribe(TopicArn=arn, Protocol='https', Endpoint=topic_sub_url)
    response = client.get_topic_attributes(TopicArn=arn)
    policy_json = '{{\n    "Version": "2008-10-17",\n    "Id": "{}/MTurkOnlyPolicy",\n    "Statement": [\n        {{\n            "Sid": "MTurkOnlyPolicy",\n            "Effect": "Allow",\n            "Principal": {{\n                "Service": "mturk-requester.amazonaws.com"\n            }},\n            "Action": "SNS:Publish",\n            "Resource": "{}"\n        }}\n    ]}}'.format(arn, arn)
    client.set_topic_attributes(TopicArn=arn, AttributeName='Policy', AttributeValue=policy_json)
    return arn