import platform
import socket
import struct
from os_ken.lib import addrconv
def sa_in4(addr, port=0):
    data = struct.pack('!H4s', port, addrconv.ipv4.text_to_bin(addr))
    hdr = _hdr(_SIN_SIZE, socket.AF_INET)
    return _pad_to(hdr + data, _SIN_SIZE)