import time
from libcloud.utils.py3 import _real_unicode as u
from libcloud.utils.xml import findall, findattr, findtext
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.aliyun import AliyunXmlResponse, SignedAliyunConnection
from libcloud.compute.types import NodeState, StorageVolumeState, VolumeSnapshotState
def ex_modify_security_group_by_id(self, group_id=None, name=None, description=None):
    """
        Modify a new security group.
        :keyword group_id: id of the security group
        :type group_id: ``str``
        :keyword name: new name of the security group
        :type name: ``unicode``
        :keyword description: new description of the security group
        :type description: ``unicode``
        """
    params = {'Action': 'ModifySecurityGroupAttribute', 'RegionId': self.region}
    if not group_id:
        raise AttributeError('group_id is required')
    params['SecurityGroupId'] = group_id
    if name:
        params['SecurityGroupName'] = name
    if description:
        params['Description'] = description
    resp = self.connection.request(self.path, params)
    return resp.success()