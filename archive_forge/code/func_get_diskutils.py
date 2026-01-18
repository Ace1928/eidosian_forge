from oslo_utils import importutils
from os_win._i18n import _  # noqa
from os_win import exceptions
from os_win.utils import hostutils
from os_win.utils.io import namedpipe
from os_win.utils import processutils
def get_diskutils():
    return _get_class(class_type='diskutils')