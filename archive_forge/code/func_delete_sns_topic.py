import boto3
import os
import json
import re
from datetime import datetime
from botocore.exceptions import ClientError
from botocore.exceptions import ProfileNotFound
def delete_sns_topic(topic_arn):
    client = boto3.client('sns', region_name='us-east-1')
    client.delete_topic(TopicArn=topic_arn)