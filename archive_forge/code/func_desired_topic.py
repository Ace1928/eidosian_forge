import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def desired_topic(module, notification_type):
    arg_dict = module.params.get(notification_type.lower() + '_notifications')
    if arg_dict:
        return arg_dict.get('topic', None)
    else:
        return None