import datetime
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_message
from ansible_collections.amazon.aws.plugins.module_utils.cloudfront_facts import CloudFrontFactsServiceManager
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def create_invalidation(self, distribution_id, invalidation_batch):
    current_invalidation_response = self.get_invalidation(distribution_id, invalidation_batch['CallerReference'])
    try:
        response = self.client.create_invalidation(DistributionId=distribution_id, InvalidationBatch=invalidation_batch)
        response.pop('ResponseMetadata', None)
        if current_invalidation_response:
            return (response, False)
        else:
            return (response, True)
    except is_boto3_error_message('Your request contains a caller reference that was used for a previous invalidation batch for the same distribution.'):
        self.module.warn('InvalidationBatch target paths are not modifiable. To make a new invalidation please update caller_reference.')
        return (current_invalidation_response, False)
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        self.module.fail_json_aws(e, msg='Error creating CloudFront invalidations.')