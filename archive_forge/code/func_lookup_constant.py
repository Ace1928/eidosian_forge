from ansible.errors import AnsibleLookupError
from ansible.plugins.lookup import LookupBase
import ansible_collections.amazon.aws.plugins.module_utils.botocore as botocore_utils
import ansible_collections.amazon.aws.plugins.module_utils.common as common_utils
def lookup_constant(self, name):
    if name == 'MINIMUM_BOTOCORE_VERSION':
        return botocore_utils.MINIMUM_BOTOCORE_VERSION
    if name == 'MINIMUM_BOTO3_VERSION':
        return botocore_utils.MINIMUM_BOTO3_VERSION
    if name == 'HAS_BOTO3':
        return botocore_utils.HAS_BOTO3
    if name == 'AMAZON_AWS_COLLECTION_VERSION':
        return common_utils.AMAZON_AWS_COLLECTION_VERSION
    if name == 'AMAZON_AWS_COLLECTION_NAME':
        return common_utils.AMAZON_AWS_COLLECTION_NAME
    if name == 'COMMUNITY_AWS_COLLECTION_VERSION':
        if not HAS_COMMUNITY:
            raise AnsibleLookupError('Unable to load ansible_collections.community.aws.plugins.module_utils.common')
        return community_utils.COMMUNITY_AWS_COLLECTION_VERSION
    if name == 'COMMUNITY_AWS_COLLECTION_NAME':
        if not HAS_COMMUNITY:
            raise AnsibleLookupError('Unable to load ansible_collections.community.aws.plugins.module_utils.common')
        return community_utils.COMMUNITY_AWS_COLLECTION_NAME