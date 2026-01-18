import base64
import struct
from os_ken.lib import addrconv
@classmethod
def register_unknown_type(cls):

    def _register_type(subcls):
        cls._UNKNOWN_TYPE = subcls
        return subcls
    return _register_type