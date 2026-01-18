from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
def byte_values_to_strings_before_compare(rules):
    for idx in range(len(rules)):
        if rules[idx].get('Statement', {}).get('ByteMatchStatement', {}).get('SearchString'):
            rules[idx]['Statement']['ByteMatchStatement']['SearchString'] = rules[idx].get('Statement').get('ByteMatchStatement').get('SearchString').decode('utf-8')
        else:
            for statement in ['AndStatement', 'OrStatement', 'NotStatement']:
                if rules[idx].get('Statement', {}).get(statement):
                    rules[idx] = nested_byte_values_to_strings(rules[idx], statement)
    return rules