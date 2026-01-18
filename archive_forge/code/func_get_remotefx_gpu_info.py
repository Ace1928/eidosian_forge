import socket
from oslo_log import log as logging
from os_win._i18n import _
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
from os_win.utils.winapi import libs as w_lib
def get_remotefx_gpu_info(self):
    """Returns information about the GPUs used for RemoteFX.

        :returns: list with dictionaries containing information about each
            GPU used for RemoteFX.
        """
    gpus = []
    all_gpus = self._conn.Msvm_Physical3dGraphicsProcessor(EnabledForVirtualization=True)
    for gpu in all_gpus:
        gpus.append({'name': gpu.Name, 'driver_version': gpu.DriverVersion, 'total_video_ram': gpu.TotalVideoMemory, 'available_video_ram': gpu.AvailableVideoMemory, 'directx_version': gpu.DirectXVersion})
    return gpus