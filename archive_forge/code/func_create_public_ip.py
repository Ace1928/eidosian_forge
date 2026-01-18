import time
from libcloud.utils.py3 import _real_unicode as u
from libcloud.utils.xml import findall, findattr, findtext
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.aliyun import AliyunXmlResponse, SignedAliyunConnection
from libcloud.compute.types import NodeState, StorageVolumeState, VolumeSnapshotState
def create_public_ip(self, instance_id):
    """
        Create public ip.

        :keyword instance_id: instance id for allocating public ip.
        :type    instance_id: ``str``

        :return public ip
        :rtype ``str``
        """
    params = {'Action': 'AllocatePublicIpAddress', 'InstanceId': instance_id}
    resp = self.connection.request(self.path, params=params)
    return findtext(resp.object, 'IpAddress', namespace=self.namespace)