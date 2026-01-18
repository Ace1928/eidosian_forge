from os_ken.lib import addrconv
import struct
def haddr_to_int(addr):
    """Convert mac address string in human readable format into
    integer value"""
    try:
        return int(addr.replace(':', ''), 16)
    except:
        raise ValueError