import contextlib
import ctypes
from oslo_log import log as logging
import six
from os_win._i18n import _
from os_win import _utils
import os_win.conf
from os_win import exceptions
from os_win.utils.storage import diskutils
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import libs as w_lib
from os_win.utils.winapi.libs import hbaapi as fc_struct
def get_fc_hba_ports(self):
    hba_ports = []
    adapter_count = self.get_fc_hba_count()
    for adapter_index in range(adapter_count):
        try:
            adapter_name = self._get_adapter_name(adapter_index)
        except Exception as exc:
            msg = 'Could not retrieve FC HBA adapter name for adapter number: %(adapter_index)s. Exception: %(exc)s'
            LOG.warning(msg, dict(adapter_index=adapter_index, exc=exc))
            continue
        try:
            hba_ports += self._get_fc_hba_adapter_ports(adapter_name)
        except Exception as exc:
            msg = 'Could not retrieve FC HBA ports for adapter: %(adapter_name)s. Exception: %(exc)s'
            LOG.warning(msg, dict(adapter_name=adapter_name, exc=exc))
    return hba_ports