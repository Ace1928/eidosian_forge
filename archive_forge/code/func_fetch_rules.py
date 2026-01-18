import datetime
import time
from copy import deepcopy
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_message
from ansible_collections.amazon.aws.plugins.module_utils.botocore import normalize_boto3_result
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def fetch_rules(client, module, name):
    try:
        current_lifecycle = client.get_bucket_lifecycle_configuration(aws_retry=True, Bucket=name)
        current_lifecycle_rules = normalize_boto3_result(current_lifecycle['Rules'])
    except is_boto3_error_code('NoSuchLifecycleConfiguration'):
        current_lifecycle_rules = []
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e)
    return current_lifecycle_rules