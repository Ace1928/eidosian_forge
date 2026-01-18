import json
import traceback
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def delete_rest_api(module, client, api_id):
    """
    Deletes entire REST API setup
    """
    try:
        delete_response = delete_api(client, api_id)
    except (botocore.exceptions.ClientError, botocore.exceptions.EndpointConnectionError) as e:
        module.fail_json_aws(e, msg=f'deleting API {api_id}')
    return delete_response