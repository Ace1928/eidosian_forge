from __future__ import (absolute_import, division, print_function)
@staticmethod
def cidr_to_netmask(cidr):
    """
        Converts a CIDR Network string to full blown IP/Subnet format in decimal format.
        Decided not use IP Address module to keep includes to a minimum.

        :param cidr: String object in CIDR format to be processed
        :type cidr: str

        :return: A string object that looks like this "x.x.x.x/y.y.y.y"
        :rtype: str
        """
    if isinstance(cidr, str):
        cidr = int(cidr)
        mask = 4294967295 >> 32 - cidr << 32 - cidr
        return str((4278190080 & mask) >> 24) + '.' + str((16711680 & mask) >> 16) + '.' + str((65280 & mask) >> 8) + '.' + str(255 & mask)