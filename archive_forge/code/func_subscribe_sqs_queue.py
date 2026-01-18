import uuid
import hashlib
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.compat import json
import boto
def subscribe_sqs_queue(self, topic, queue):
    """
        Subscribe an SQS queue to a topic.

        This is convenience method that handles most of the complexity involved
        in using an SQS queue as an endpoint for an SNS topic.  To achieve this
        the following operations are performed:

        * The correct ARN is constructed for the SQS queue and that ARN is
          then subscribed to the topic.
        * A JSON policy document is contructed that grants permission to
          the SNS topic to send messages to the SQS queue.
        * This JSON policy is then associated with the SQS queue using
          the queue's set_attribute method.  If the queue already has
          a policy associated with it, this process will add a Statement to
          that policy.  If no policy exists, a new policy will be created.

        :type topic: string
        :param topic: The ARN of the new topic.

        :type queue: A boto Queue object
        :param queue: The queue you wish to subscribe to the SNS Topic.
        """
    t = queue.id.split('/')
    q_arn = queue.arn
    sid = hashlib.md5((topic + q_arn).encode('utf-8')).hexdigest()
    sid_exists = False
    resp = self.subscribe(topic, 'sqs', q_arn)
    attr = queue.get_attributes('Policy')
    if 'Policy' in attr:
        policy = json.loads(attr['Policy'])
    else:
        policy = {}
    if 'Version' not in policy:
        policy['Version'] = '2008-10-17'
    if 'Statement' not in policy:
        policy['Statement'] = []
    for s in policy['Statement']:
        if s['Sid'] == sid:
            sid_exists = True
    if not sid_exists:
        statement = {'Action': 'SQS:SendMessage', 'Effect': 'Allow', 'Principal': {'AWS': '*'}, 'Resource': q_arn, 'Sid': sid, 'Condition': {'StringLike': {'aws:SourceArn': topic}}}
        policy['Statement'].append(statement)
    queue.set_attribute('Policy', json.dumps(policy))
    return resp