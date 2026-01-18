import time
from libcloud.utils.py3 import _real_unicode as u
from libcloud.utils.xml import findall, findattr, findtext
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.aliyun import AliyunXmlResponse, SignedAliyunConnection
from libcloud.compute.types import NodeState, StorageVolumeState, VolumeSnapshotState
def ex_list_security_group_attributes(self, group_id=None, nic_type='internet'):
    """
        List security group attributes in the current region.

        :keyword group_id: security group id.
        :type group_id: ``str``

        :keyword nic_type: internet|intranet.
        :type nic_type: ``str``

        :return: a list of defined security group Attributes
        :rtype: ``list`` of ``ECSSecurityGroupAttribute``
        """
    params = {'Action': 'DescribeSecurityGroupAttribute', 'RegionId': self.region, 'NicType': nic_type}
    if group_id is None:
        raise AttributeError('group_id is required')
    params['SecurityGroupId'] = group_id
    resp_object = self.connection.request(self.path, params).object
    sga_elements = findall(resp_object, 'Permissions/Permission', namespace=self.namespace)
    return [self._to_security_group_attribute(el) for el in sga_elements]