import functools
import time
import uuid
from eventlet import patcher
from eventlet import tpool
from oslo_log import log as logging
from oslo_utils import uuidutils
from six.moves import range  # noqa
from os_win._i18n import _
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils import baseutils
from os_win.utils import jobutils
from os_win.utils import pathutils
def enable_secure_boot(self, vm_name, msft_ca_required):
    """Enables Secure Boot for the instance with the given name.

        :param vm_name: The name of the VM for which Secure Boot will be
                        enabled.
        :param msft_ca_required: boolean specifying whether the VM will
                                 require Microsoft UEFI Certificate
                                 Authority for Secure Boot. Only Linux
                                 guests require this CA.
        """
    vs_data = self._lookup_vm_check(vm_name)
    self._set_secure_boot(vs_data, msft_ca_required)
    self._modify_virtual_system(vs_data)