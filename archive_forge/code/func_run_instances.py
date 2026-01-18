import json
import logging
from aliyunsdkcore import client
from aliyunsdkcore.acs_exception.exceptions import ClientException, ServerException
from aliyunsdkecs.request.v20140526.AllocatePublicIpAddressRequest import (
from aliyunsdkecs.request.v20140526.AuthorizeSecurityGroupRequest import (
from aliyunsdkecs.request.v20140526.CreateInstanceRequest import CreateInstanceRequest
from aliyunsdkecs.request.v20140526.CreateKeyPairRequest import CreateKeyPairRequest
from aliyunsdkecs.request.v20140526.CreateSecurityGroupRequest import (
from aliyunsdkecs.request.v20140526.CreateVpcRequest import CreateVpcRequest
from aliyunsdkecs.request.v20140526.CreateVSwitchRequest import CreateVSwitchRequest
from aliyunsdkecs.request.v20140526.DeleteInstanceRequest import DeleteInstanceRequest
from aliyunsdkecs.request.v20140526.DeleteInstancesRequest import DeleteInstancesRequest
from aliyunsdkecs.request.v20140526.DeleteKeyPairsRequest import DeleteKeyPairsRequest
from aliyunsdkecs.request.v20140526.DescribeInstancesRequest import (
from aliyunsdkecs.request.v20140526.DescribeKeyPairsRequest import (
from aliyunsdkecs.request.v20140526.DescribeSecurityGroupsRequest import (
from aliyunsdkecs.request.v20140526.DescribeVpcsRequest import DescribeVpcsRequest
from aliyunsdkecs.request.v20140526.DescribeVSwitchesRequest import (
from aliyunsdkecs.request.v20140526.ImportKeyPairRequest import ImportKeyPairRequest
from aliyunsdkecs.request.v20140526.RunInstancesRequest import RunInstancesRequest
from aliyunsdkecs.request.v20140526.StartInstanceRequest import StartInstanceRequest
from aliyunsdkecs.request.v20140526.StopInstanceRequest import StopInstanceRequest
from aliyunsdkecs.request.v20140526.StopInstancesRequest import StopInstancesRequest
from aliyunsdkecs.request.v20140526.TagResourcesRequest import TagResourcesRequest
def run_instances(self, instance_type, image_id, tags, security_group_id, vswitch_id, key_pair_name, amount=1, optimized='optimized', instance_charge_type='PostPaid', spot_strategy='SpotWithPriceLimit', internet_charge_type='PayByTraffic', internet_max_bandwidth_out=1):
    """Create one or more pay-as-you-go or subscription
            Elastic Compute Service (ECS) instances

        :param instance_type: The instance type of the ECS.
        :param image_id: The ID of the image used to create the instance.
        :param tags: The tags of the instance.
        :param security_group_id: The ID of the security group to which to
                                  assign the instance. Instances in the same
                                  security group can communicate with
                                  each other.
        :param vswitch_id: The ID of the vSwitch to which to connect
                           the instance.
        :param key_pair_name: The name of the key pair to be bound to
                              the instance.
        :param amount: The number of instances that you want to create.
        :param optimized: Specifies whether the instance is I/O optimized
        :param instance_charge_type: The billing method of the instance.
                                     Default value: PostPaid.
        :param spot_strategy: The preemption policy for the pay-as-you-go
                              instance.
        :param internet_charge_type: The billing method for network usage.
                                     Default value: PayByTraffic.
        :param internet_max_bandwidth_out: The maximum inbound public
                                           bandwidth. Unit: Mbit/s.
        :return: The created instance IDs.
        """
    request = RunInstancesRequest()
    request.set_InstanceType(instance_type)
    request.set_ImageId(image_id)
    request.set_IoOptimized(optimized)
    request.set_InstanceChargeType(instance_charge_type)
    request.set_SpotStrategy(spot_strategy)
    request.set_InternetChargeType(internet_charge_type)
    request.set_InternetMaxBandwidthOut(internet_max_bandwidth_out)
    request.set_Tags(tags)
    request.set_Amount(amount)
    request.set_SecurityGroupId(security_group_id)
    request.set_VSwitchId(vswitch_id)
    request.set_KeyPairName(key_pair_name)
    response = self._send_request(request)
    if response is not None:
        instance_ids = response.get('InstanceIdSets').get('InstanceIdSet')
        return instance_ids
    logging.error('instance created failed.')
    return None