from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def filter_findings(findings, type_filter):
    if not type_filter:
        return findings
    filter_map = dict(error='ERROR', security='SECURITY_WARNING', suggestion='SUGGESTION', warning='WARNING')
    allowed_types = [filter_map[t] for t in type_filter]
    filtered_results = [f for f in findings if f.get('findingType', None) in allowed_types]
    return filtered_results