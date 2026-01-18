from copy import deepcopy
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.community.aws.plugins.module_utils.ec2 import BaseEc2Manager
from ansible_collections.community.aws.plugins.module_utils.ec2 import Boto3Mixin
from ansible_collections.community.aws.plugins.module_utils.ec2 import Ec2WaiterFactory
def set_subnets(self, subnets=None, purge=True):
    if subnets is None:
        return False
    current_subnets = set(self._preupdate_resource.get('SubnetIds', []))
    desired_subnets = set(subnets)
    if not purge:
        desired_subnets = desired_subnets.union(current_subnets)
    subnet_details = self._describe_subnets(SubnetIds=list(desired_subnets))
    vpc_id = self.subnets_to_vpc(desired_subnets, subnet_details)
    self._set_resource_value('VpcId', vpc_id, immutable=True)
    azs = [s.get('AvailabilityZoneId') for s in subnet_details]
    if len(azs) != len(set(azs)):
        self.module.fail_json(msg='Only one attachment subnet per availability zone may be set.', availability_zones=azs, subnets=subnet_details)
    subnets_to_add = list(desired_subnets.difference(current_subnets))
    subnets_to_remove = list(current_subnets.difference(desired_subnets))
    if not subnets_to_remove and (not subnets_to_add):
        return False
    self._subnet_updates = dict(add=subnets_to_add, remove=subnets_to_remove)
    self._set_resource_value('SubnetIds', list(desired_subnets))
    return True