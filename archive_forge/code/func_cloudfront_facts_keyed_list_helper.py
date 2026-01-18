from functools import partial
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from .retries import AWSRetry
from .tagging import boto3_tag_list_to_ansible_dict
def cloudfront_facts_keyed_list_helper(list_to_key):
    result = dict()
    for item in list_to_key:
        distribution_id = item['Id']
        if 'Items' in item['Aliases']:
            result.update({alias: item for alias in item['Aliases']['Items']})
        result.update({distribution_id: item})
    return result