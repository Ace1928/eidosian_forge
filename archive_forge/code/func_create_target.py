from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.backup.base import BackupDriver, BackupTarget, BackupTargetJob
from libcloud.backup.types import Provider, BackupTargetType
from libcloud.common.dimensiondata import (
def create_target(self, name, address, type=BackupTargetType.VIRTUAL, extra=None):
    """
        Creates a new backup target

        :param name: Name of the target (not used)
        :type name: ``str``

        :param address: The ID of the node in Dimension Data Cloud
        :type address: ``str``

        :param type: Backup target type, only Virtual supported
        :type type: :class:`BackupTargetType`

        :param extra: (optional) Extra attributes (driver specific).
        :type extra: ``dict``

        :rtype: Instance of :class:`BackupTarget`
        """
    if extra is not None:
        service_plan = extra.get('servicePlan', DEFAULT_BACKUP_PLAN)
    else:
        service_plan = DEFAULT_BACKUP_PLAN
        extra = {'servicePlan': service_plan}
    create_node = ET.Element('NewBackup', {'xmlns': BACKUP_NS})
    create_node.set('servicePlan', service_plan)
    response = self.connection.request_with_orgId_api_1('server/%s/backup' % address, method='POST', data=ET.tostring(create_node)).object
    asset_id = None
    for info in findall(response, 'additionalInformation', GENERAL_NS):
        if info.get('name') == 'assetId':
            asset_id = findtext(info, 'value', GENERAL_NS)
    return BackupTarget(id=asset_id, name=name, address=address, type=type, extra=extra, driver=self)