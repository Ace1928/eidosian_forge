import datetime
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
@staticmethod
def insert_quantities(dict_with_items):
    if 'Items' in dict_with_items and isinstance(dict_with_items['Items'], list):
        dict_with_items['Quantity'] = len(dict_with_items['Items'])
    for k, v in dict_with_items.items():
        if isinstance(v, dict) and 'Items' in v:
            v['Quantity'] = len(v['Items'])
    return dict_with_items