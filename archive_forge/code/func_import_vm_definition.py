from oslo_log import log as logging
from os_win._i18n import _
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
from os_win.utils.compute import vmutils
from os_win.utils import jobutils
def import_vm_definition(self, export_config_file_path, snapshot_folder_path, new_uuid=False):
    ref, job_path, ret_val = self._vs_man_svc.ImportSystemDefinition(new_uuid, snapshot_folder_path, export_config_file_path)
    self._jobutils.check_ret_val(ret_val, job_path)