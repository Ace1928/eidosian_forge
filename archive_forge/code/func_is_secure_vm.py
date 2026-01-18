import re
from oslo_log import log as logging
import six
from os_win._i18n import _
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
from oslo_utils import units
def is_secure_vm(self, instance_name):
    inst_id = self.get_vm_id(instance_name)
    security_profile = _wqlutils.get_element_associated_class(self._conn, self._SECURITY_SETTING_DATA, element_uuid=inst_id)
    if security_profile:
        return security_profile[0].EncryptStateAndVmMigrationTraffic
    return False