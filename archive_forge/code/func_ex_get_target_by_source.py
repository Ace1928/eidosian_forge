from libcloud.backup.base import (
from libcloud.backup.types import BackupTargetType, BackupTargetJobStatusType
from libcloud.common.google import GoogleResponse, GoogleBaseConnection
from libcloud.utils.iso8601 import parse_date
def ex_get_target_by_source(self, source):
    return BackupTarget(id=source, name=source, address=source, type=BackupTargetType.VOLUME, driver=self.connection.driver, extra={'source': source})