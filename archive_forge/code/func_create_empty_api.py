import json
import traceback
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def create_empty_api(module, client, name, endpoint_type, tags):
    """
    creates a new empty API ready to be configured. The description is
    temporarily set to show the API as incomplete but should be
    updated when the API is configured.
    """
    desc = 'Incomplete API creation by ansible api_gateway module'
    try:
        rest_api_name = name or 'ansible-temp-api'
        awsret = create_api(client, name=rest_api_name, description=desc, endpoint_type=endpoint_type, tags=tags)
    except (botocore.exceptions.ClientError, botocore.exceptions.EndpointConnectionError) as e:
        module.fail_json_aws(e, msg='creating API')
    return awsret['id']