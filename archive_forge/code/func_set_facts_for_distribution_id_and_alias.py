from ansible_collections.amazon.aws.plugins.module_utils.cloudfront_facts import CloudFrontFactsServiceManager
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def set_facts_for_distribution_id_and_alias(details, facts, distribution_id, aliases):
    facts[distribution_id] = details
    facts['result'] = details
    facts['result']['DistributionId'] = distribution_id
    for alias in aliases:
        facts[alias] = details
    return facts