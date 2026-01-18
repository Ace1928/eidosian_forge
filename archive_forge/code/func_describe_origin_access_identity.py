import datetime
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.cloudfront_facts import CloudFrontFactsServiceManager
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def describe_origin_access_identity(self, origin_access_identity_id, fail_if_missing=True):
    try:
        return self.__cloudfront_facts_mgr.get_origin_access_identity(id=origin_access_identity_id, fail_if_error=False)
    except is_boto3_error_code('NoSuchCloudFrontOriginAccessIdentity') as e:
        if fail_if_missing:
            self.module.fail_json_aws(e, msg='Error getting etag from origin_access_identity.')
        return {}
    except (ClientError, BotoCoreError) as e:
        self.module.fail_json_aws(e, msg='Error getting etag from origin_access_identity.')