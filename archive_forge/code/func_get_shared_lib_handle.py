import importlib
from os_win._i18n import _
from os_win import exceptions
def get_shared_lib_handle(lib_name):
    module = _get_shared_lib_module(lib_name)
    return module.lib_handle