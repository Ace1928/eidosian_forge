from libcloud.utils.xml import findall, findtext
from libcloud.common.aws import AWSGenericResponse, SignedAWSConnection
from libcloud.backup.base import (
from libcloud.backup.types import BackupTargetType, BackupTargetJobStatusType
from libcloud.utils.iso8601 import parse_date
def ex_get_target_by_volume_id(self, volume_id):
    return BackupTarget(id=volume_id, name=volume_id, address=volume_id, type=BackupTargetType.VOLUME, driver=self.connection.driver, extra={'volume-id': volume_id})