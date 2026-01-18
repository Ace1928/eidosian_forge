from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
def ensure_igw_present(self, igw_id, vpc_id, tags, purge_tags, force_attach, detach_vpc):
    igw = None
    if igw_id:
        igw = self.get_matching_igw(None, gateway_id=igw_id)
    elif vpc_id:
        igw = self.get_matching_igw(vpc_id)
    if igw is None:
        if self._check_mode:
            self._results['changed'] = True
            self._results['gateway_id'] = None
            return self._results
        if vpc_id:
            self.get_matching_vpc(vpc_id)
        try:
            create_params = {}
            if tags:
                create_params['TagSpecifications'] = boto3_tag_specifications(tags, types='internet-gateway')
            response = self._connection.create_internet_gateway(aws_retry=True, **create_params)
            waiter = get_waiter(self._connection, 'internet_gateway_exists')
            waiter.wait(InternetGatewayIds=[response['InternetGateway']['InternetGatewayId']])
            self._results['changed'] = True
            igw = camel_dict_to_snake_dict(response['InternetGateway'])
            if vpc_id:
                self.attach_vpc(igw['internet_gateway_id'], vpc_id)
        except botocore.exceptions.WaiterError as e:
            self._module.fail_json_aws(e, msg='No Internet Gateway exists.')
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            self._module.fail_json_aws(e, msg='Unable to create Internet Gateway')
    else:
        igw_vpc_id = None
        if len(igw['attachments']) > 0:
            igw_vpc_id = igw['attachments'][0]['vpc_id']
            if detach_vpc:
                if self._check_mode:
                    self._results['changed'] = True
                    self._results['gateway_id'] = igw['internet_gateway_id']
                    return self._results
                self.detach_vpc(igw['internet_gateway_id'], igw_vpc_id)
            elif igw_vpc_id != vpc_id:
                if self._check_mode:
                    self._results['changed'] = True
                    self._results['gateway_id'] = igw['internet_gateway_id']
                    return self._results
                if force_attach:
                    self.get_matching_vpc(vpc_id)
                    self.detach_vpc(igw['internet_gateway_id'], igw_vpc_id)
                    self.attach_vpc(igw['internet_gateway_id'], vpc_id)
                else:
                    self._module.fail_json(msg='VPC already attached, but does not match requested VPC.')
        elif vpc_id:
            if self._check_mode:
                self._results['changed'] = True
                self._results['gateway_id'] = igw['internet_gateway_id']
                return self._results
            self.get_matching_vpc(vpc_id)
            self.attach_vpc(igw['internet_gateway_id'], vpc_id)
    self._results['changed'] |= ensure_ec2_tags(self._connection, self._module, igw['internet_gateway_id'], resource_type='internet-gateway', tags=tags, purge_tags=purge_tags, retry_codes='InvalidInternetGatewayID.NotFound')
    igw = self.get_matching_igw(vpc_id, gateway_id=igw['internet_gateway_id'])
    igw_info = self.get_igw_info(igw, vpc_id)
    self._results.update(igw_info)
    return self._results