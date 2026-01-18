from collections import namedtuple
from time import sleep
from ansible.module_utils._text import to_text
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from .retries import AWSRetry
from .tagging import ansible_dict_to_boto3_tag_list
from .tagging import boto3_tag_list_to_ansible_dict
from .tagging import compare_aws_tags
from .waiters import get_waiter
def arg_spec_to_rds_params(options_dict):
    tags = options_dict.pop('tags')
    has_processor_features = False
    if 'processor_features' in options_dict:
        has_processor_features = True
        processor_features = options_dict.pop('processor_features')
    camel_options = snake_dict_to_camel_dict(options_dict, capitalize_first=True)
    for key in list(camel_options.keys()):
        for old, new in (('Db', 'DB'), ('Iam', 'IAM'), ('Az', 'AZ'), ('Ca', 'CA')):
            if old in key:
                camel_options[key.replace(old, new)] = camel_options.pop(key)
    camel_options['Tags'] = tags
    if has_processor_features:
        camel_options['ProcessorFeatures'] = processor_features
    return camel_options