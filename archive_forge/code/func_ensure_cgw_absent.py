from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
@AWSRetry.jittered_backoff(delay=2, max_delay=30, retries=6, catch_extra_error_codes=['IncorrectState'])
def ensure_cgw_absent(self, gw_id):
    response = self.ec2.delete_customer_gateway(DryRun=False, CustomerGatewayId=gw_id)
    return response