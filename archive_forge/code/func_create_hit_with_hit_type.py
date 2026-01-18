import boto3
import os
import json
import re
from datetime import datetime
from botocore.exceptions import ClientError
from botocore.exceptions import ProfileNotFound
def create_hit_with_hit_type(opt, page_url, hit_type_id, num_assignments, is_sandbox):
    """
    Creates the actual HIT given the type and page to direct clients to.
    """
    page_url = page_url.replace('&', '&amp;')
    amazon_ext_url = 'http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2006-07-14/ExternalQuestion.xsd'
    question_data_struture = '<ExternalQuestion xmlns="{}"><ExternalURL>{}</ExternalURL><FrameHeight>{}</FrameHeight></ExternalQuestion>'.format(amazon_ext_url, page_url, opt.get('frame_height', 650))
    client = get_mturk_client(is_sandbox)
    response = client.create_hit_with_hit_type(HITTypeId=hit_type_id, MaxAssignments=num_assignments, LifetimeInSeconds=31536000, Question=question_data_struture)
    hit_type_id = response['HIT']['HITTypeId']
    hit_id = response['HIT']['HITId']
    url_target = 'workersandbox'
    if not is_sandbox:
        url_target = 'www'
    hit_link = 'https://{}.mturk.com/mturk/preview?groupId={}'.format(url_target, hit_type_id)
    return (hit_link, hit_id, response)