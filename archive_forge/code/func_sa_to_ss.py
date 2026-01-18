import platform
import socket
import struct
from os_ken.lib import addrconv
def sa_to_ss(sa):
    return _pad_to(sa, _SS_MAXSIZE)