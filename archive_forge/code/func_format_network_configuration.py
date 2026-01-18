from ansible_collections.amazon.aws.plugins.module_utils.ec2 import get_ec2_security_group_ids_from_names
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def format_network_configuration(self, network_config):
    result = dict()
    if 'subnets' in network_config:
        result['subnets'] = network_config['subnets']
    else:
        self.module.fail_json(msg='Network configuration must include subnets')
    if 'security_groups' in network_config:
        groups = network_config['security_groups']
        if any((not sg.startswith('sg-') for sg in groups)):
            try:
                vpc_id = self.ec2.describe_subnets(SubnetIds=[result['subnets'][0]])['Subnets'][0]['VpcId']
                groups = get_ec2_security_group_ids_from_names(groups, self.ec2, vpc_id)
            except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                self.module.fail_json_aws(e, msg="Couldn't look up security groups")
        result['securityGroups'] = groups
    if 'assign_public_ip' in network_config:
        if network_config['assign_public_ip'] is True:
            result['assignPublicIp'] = 'ENABLED'
        else:
            result['assignPublicIp'] = 'DISABLED'
    return dict(awsvpcConfiguration=result)