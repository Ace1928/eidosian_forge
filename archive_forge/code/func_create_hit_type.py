import boto3
import os
import json
import re
from datetime import datetime
from botocore.exceptions import ClientError
from botocore.exceptions import ProfileNotFound
def create_hit_type(hit_title, hit_description, hit_keywords, hit_reward, assignment_duration_in_seconds, is_sandbox, qualifications=None, auto_approve_delay=7 * 24 * 3600):
    """
    Create a HIT type to be used to generate HITs of the requested params.
    """
    client = get_mturk_client(is_sandbox)
    locale_requirements = []
    has_locale_qual = False
    if qualifications is not None:
        for q in qualifications:
            if q['QualificationTypeId'] == '00000000000000000071':
                has_locale_qual = True
        locale_requirements += qualifications
    if not has_locale_qual:
        locale_requirements.append({'QualificationTypeId': '00000000000000000071', 'Comparator': 'In', 'LocaleValues': [{'Country': 'US'}, {'Country': 'CA'}, {'Country': 'GB'}, {'Country': 'AU'}, {'Country': 'NZ'}], 'RequiredToPreview': True})
    response = client.create_hit_type(AutoApprovalDelayInSeconds=auto_approve_delay, AssignmentDurationInSeconds=assignment_duration_in_seconds, Reward=str(hit_reward), Title=hit_title, Keywords=hit_keywords, Description=hit_description, QualificationRequirements=locale_requirements)
    hit_type_id = response['HITTypeId']
    return hit_type_id