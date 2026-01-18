from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
def ensure_igw_absent(self, igw_id, vpc_id):
    igw = self.get_matching_igw(vpc_id, gateway_id=igw_id)
    if igw is None:
        return self._results
    igw_vpc_id = ''
    if len(igw['attachments']) > 0:
        igw_vpc_id = igw['attachments'][0]['vpc_id']
    if vpc_id and igw_vpc_id != vpc_id:
        self._module.fail_json(msg=f'Supplied VPC ({vpc_id}) does not match found VPC ({igw_vpc_id}), aborting')
    if self._check_mode:
        self._results['changed'] = True
        return self._results
    try:
        self._results['changed'] = True
        if igw_vpc_id:
            self.detach_vpc(igw['internet_gateway_id'], igw_vpc_id)
        self._connection.delete_internet_gateway(aws_retry=True, InternetGatewayId=igw['internet_gateway_id'])
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        self._module.fail_json_aws(e, msg='Unable to delete Internet Gateway')
    return self._results