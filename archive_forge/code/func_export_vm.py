from oslo_log import log as logging
from os_win._i18n import _
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
from os_win.utils.compute import vmutils
from os_win.utils import jobutils
def export_vm(self, vm_name, export_path, copy_snapshots_config=constants.EXPORT_CONFIG_SNAPSHOTS_ALL, copy_vm_storage=False, create_export_subdir=False):
    vm = self._vmutils._lookup_vm(vm_name)
    export_setting_data = self._get_export_setting_data(vm_name)
    export_setting_data.CopySnapshotConfiguration = copy_snapshots_config
    export_setting_data.CopyVmStorage = copy_vm_storage
    export_setting_data.CreateVmExportSubdirectory = create_export_subdir
    job_path, ret_val = self._vs_man_svc.ExportSystemDefinition(ComputerSystem=vm.path_(), ExportDirectory=export_path, ExportSettingData=export_setting_data.GetText_(1))
    self._jobutils.check_ret_val(ret_val, job_path)