from functools import partial
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from .retries import AWSRetry
from .tagging import boto3_tag_list_to_ansible_dict
def describe_cloudfront_property(self, client_method, error, post_process, **kwargs):
    fail_if_error = kwargs.pop('fail_if_error', True)
    try:
        method = getattr(self.client, client_method)
        api_kwargs = snake_dict_to_camel_dict(kwargs, capitalize_first=True)
        result = method(aws_retry=True, **api_kwargs)
        result.pop('ResponseMetadata', None)
        if post_process:
            result = post_process(result)
        return result
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        if not fail_if_error:
            raise
        self.module.fail_json_aws(e, msg=error)