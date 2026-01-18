from os_ken.lib import addrconv
import struct
def haddr_bitand(addr, mask):
    return b''.join((struct.Struct('>B').pack(int(a) & int(m)) for a, m in zip(addr, mask)))