from oslo_log import log as logging
from os_win._i18n import _
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils import baseutils
def enable_vm_metrics_collection(self, vm_name):
    vm = self._get_vm(vm_name)
    disks = self._get_vm_resources(vm_name, self._STORAGE_ALLOC_SETTING_DATA_CLASS)
    filtered_disks = [d for d in disks if d.ResourceSubType != self._DVD_DISK_RES_SUB_TYPE]
    for disk in filtered_disks:
        self._enable_metrics(disk)
    metrics_names = [self._CPU_METRICS, self._MEMORY_METRICS]
    self._enable_metrics(vm, metrics_names)