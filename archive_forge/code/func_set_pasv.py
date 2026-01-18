import sys
import socket
from socket import _GLOBAL_DEFAULT_TIMEOUT
def set_pasv(self, val):
    """Use passive or active mode for data transfers.
        With a false argument, use the normal PORT mode,
        With a true argument, use the PASV command."""
    self.passiveserver = val