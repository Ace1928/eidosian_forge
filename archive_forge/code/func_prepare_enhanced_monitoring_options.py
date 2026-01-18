import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def prepare_enhanced_monitoring_options(module):
    m_params = {}
    m_params['EnhancedMonitoring'] = module.params['enhanced_monitoring'] or 'DEFAULT'
    return m_params