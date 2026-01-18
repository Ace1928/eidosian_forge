from pyudev._ctypeslib.libudev import ERROR_CHECKERS, SIGNATURES
from pyudev._ctypeslib.utils import load_ctypes_library
from pyudev._errors import DeviceNotFoundAtPathError
from pyudev._util import (
from pyudev.device import Devices
def match_is_initialized(self):
    """
        Include only devices, which are initialized.

        Initialized devices have properly set device node permissions and
        context, and are (in case of network devices) fully renamed.

        Currently this will not affect devices which do not have device nodes
        and are not network interfaces.

        Return the instance again.

        .. seealso:: :attr:`Device.is_initialized`

        .. udevversion:: 165

        .. versionadded:: 0.8
        """
    self._libudev.udev_enumerate_add_match_is_initialized(self)
    return self