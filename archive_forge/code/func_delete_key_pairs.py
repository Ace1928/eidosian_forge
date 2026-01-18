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
def delete_key_pairs(self, key_pair_names):
    """Delete one or more SSH key pairs.

        :param key_pair_names: The name of the key pair.
        :return:
        """
    request = DeleteKeyPairsRequest()
    request.set_KeyPairNames(key_pair_names)
    self._send_request(request)