import json
import time
from ansible.module_utils.basic import to_text
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible.module_utils.six import string_types
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.policy import compare_policies
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.s3 import s3_extra_params
from ansible_collections.amazon.aws.plugins.module_utils.s3 import validate_bucket_name
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
def destroy_bucket(s3_client, module):
    force = module.params.get('force')
    name = module.params.get('name')
    try:
        bucket_is_present = bucket_exists(s3_client, name)
    except botocore.exceptions.EndpointConnectionError as e:
        module.fail_json_aws(e, msg=f'Invalid endpoint provided: {to_text(e)}')
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
        module.fail_json_aws(e, msg='Failed to check bucket presence')
    if not bucket_is_present:
        module.exit_json(changed=False)
    if force:
        try:
            for key_version_pairs in paginated_versions_list(s3_client, Bucket=name):
                formatted_keys = [{'Key': key, 'VersionId': version} for key, version in key_version_pairs]
                for fk in formatted_keys:
                    if not fk.get('VersionId') or fk.get('VersionId') == 'null':
                        fk.pop('VersionId')
                if formatted_keys:
                    resp = s3_client.delete_objects(Bucket=name, Delete={'Objects': formatted_keys})
                    if resp.get('Errors'):
                        objects_to_delete = ', '.join([k['Key'] for k in resp['Errors']])
                        module.fail_json(msg=f'Could not empty bucket before deleting. Could not delete objects: {objects_to_delete}', errors=resp['Errors'], response=resp)
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            module.fail_json_aws(e, msg='Failed while deleting bucket')
    try:
        delete_bucket(s3_client, name)
        s3_client.get_waiter('bucket_not_exists').wait(Bucket=name, WaiterConfig=dict(Delay=5, MaxAttempts=60))
    except botocore.exceptions.WaiterError as e:
        module.fail_json_aws(e, msg='An error occurred waiting for the bucket to be deleted.')
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
        module.fail_json_aws(e, msg='Failed to delete bucket')
    module.exit_json(changed=True)