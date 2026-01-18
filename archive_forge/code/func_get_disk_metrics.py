from oslo_log import log as logging
from os_win._i18n import _
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils import baseutils
def get_disk_metrics(self, vm_name):
    metrics_def_r = self._metrics_defs[self._DISK_RD_METRICS]
    metrics_def_w = self._metrics_defs[self._DISK_WR_METRICS]
    disks = self._get_vm_resources(vm_name, self._STORAGE_ALLOC_SETTING_DATA_CLASS)
    for disk in disks:
        metrics_values = self._get_metrics_values(disk, [metrics_def_r, metrics_def_w])
        yield {'read_mb': metrics_values[0], 'write_mb': metrics_values[1], 'instance_id': disk.InstanceID, 'host_resource': disk.HostResource[0]}