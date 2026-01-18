import time
import uuid
from collections import namedtuple
from ansible.module_utils._text import to_native
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible.module_utils.six import string_types
from ansible_collections.amazon.aws.plugins.module_utils.arn import validate_aws_arn
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_message
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import get_ec2_security_group_ids_from_names
from ansible_collections.amazon.aws.plugins.module_utils.exceptions import AnsibleAWSError
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.tower import tower_callback_script
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
def add_or_update_instance_profile(instance, desired_profile_name):
    instance_profile_setting = instance.get('IamInstanceProfile')
    if instance_profile_setting and desired_profile_name:
        if desired_profile_name in (instance_profile_setting.get('Name'), instance_profile_setting.get('Arn')):
            return False
        else:
            desired_arn = determine_iam_role(desired_profile_name)
            if instance_profile_setting.get('Arn') == desired_arn:
                return False
        try:
            association = client.describe_iam_instance_profile_associations(aws_retry=True, Filters=[{'Name': 'instance-id', 'Values': [instance['InstanceId']]}])
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            module.fail_json_aws(e, 'Could not find instance profile association')
        try:
            client.replace_iam_instance_profile_association(aws_retry=True, AssociationId=association['IamInstanceProfileAssociations'][0]['AssociationId'], IamInstanceProfile={'Arn': determine_iam_role(desired_profile_name)})
            return True
        except botocore.exceptions.ClientError as e:
            module.fail_json_aws(e, 'Could not associate instance profile')
    if not instance_profile_setting and desired_profile_name:
        try:
            client.associate_iam_instance_profile(aws_retry=True, IamInstanceProfile={'Arn': determine_iam_role(desired_profile_name)}, InstanceId=instance['InstanceId'])
            return True
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            module.fail_json_aws(e, 'Could not associate new instance profile')
    return False