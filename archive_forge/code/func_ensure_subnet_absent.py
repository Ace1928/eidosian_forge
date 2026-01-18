import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.arn import is_outpost_arn
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_tag_filter_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
def ensure_subnet_absent(conn, module):
    subnet = get_matching_subnet(conn, module, module.params['vpc_id'], module.params['cidr'])
    if subnet is None:
        return {'changed': False}
    try:
        if not module.check_mode:
            conn.delete_subnet(aws_retry=True, SubnetId=subnet['id'])
            if module.params['wait']:
                handle_waiter(conn, module, 'subnet_deleted', {'SubnetIds': [subnet['id']]}, time.time())
        return {'changed': True}
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg="Couldn't delete subnet")