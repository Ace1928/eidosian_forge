from ansible.errors import AnsibleLookupError
from ansible.module_utils._text import to_native
from ansible.module_utils.six import string_types
from ansible.utils.display import Display
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.plugin_utils.lookup import AWSLookupBase
def get_parameter_value(self, client, ssm_dict, term, on_missing, on_denied):
    ssm_dict['Name'] = term
    try:
        response = client.get_parameter(aws_retry=True, **ssm_dict)
        return response['Parameter']['Value']
    except is_boto3_error_code('ParameterNotFound'):
        if on_missing == 'error':
            raise AnsibleLookupError(f'Failed to find SSM parameter {term} (ResourceNotFound)')
        elif on_missing == 'warn':
            self.warn(f'Skipping, did not find SSM parameter {term}')
    except is_boto3_error_code('AccessDeniedException'):
        if on_denied == 'error':
            raise AnsibleLookupError(f'Failed to access SSM parameter {term} (AccessDenied)')
        elif on_denied == 'warn':
            self.warn(f'Skipping, access denied for SSM parameter {term}')
    except botocore.exceptions.ClientError as e:
        raise AnsibleLookupError(f'SSM lookup exception: {to_native(e)}')
    return None