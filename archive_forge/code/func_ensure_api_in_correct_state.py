import json
import traceback
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def ensure_api_in_correct_state(module, client, api_id, api_data):
    """Make sure that we have the API configured and deployed as instructed.

    This function first configures the API correctly uploading the
    swagger definitions and then deploys those.  Configuration and
    deployment should be closely tied because there is only one set of
    definitions so if we stop, they may be updated by someone else and
    then we deploy the wrong configuration.
    """
    configure_response = None
    try:
        configure_response = configure_api(client, api_id, api_data=api_data)
        configure_response.pop('ResponseMetadata', None)
    except (botocore.exceptions.ClientError, botocore.exceptions.EndpointConnectionError) as e:
        module.fail_json_aws(e, msg=f'configuring API {api_id}')
    deploy_response = None
    stage = module.params.get('stage')
    if stage:
        try:
            deploy_response = create_deployment(client, api_id, **module.params)
            deploy_response.pop('ResponseMetadata', None)
        except (botocore.exceptions.ClientError, botocore.exceptions.EndpointConnectionError) as e:
            msg = f'deploying api {api_id} to stage {stage}'
            module.fail_json_aws(e, msg)
    return (configure_response, deploy_response)