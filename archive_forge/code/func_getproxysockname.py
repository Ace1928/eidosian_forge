import base64
import socket
import struct
import sys
def getproxysockname(self):
    """getsockname() -> address info
        Returns the bound IP address and port number at the proxy.
        """
    return self.__proxysockname