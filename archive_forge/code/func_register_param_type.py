import abc
import struct
from os_ken.lib import addrconv
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
@staticmethod
def register_param_type(*args):

    def _register_param_type(cls):
        cause_restart_with_new_addr._RECOGNIZED_PARAMS[cls.param_type()] = cls
        return cls
    return _register_param_type(args[0])