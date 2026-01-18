import sys, re, curl, exceptions
from the command line first, then standard input.
def set_DHCP(self, flag):
    if flag:
        self.actions.append('DHCP.htm', 'dhcpStatus', 'Enable')
    else:
        self.actions.append('DHCP.htm', 'dhcpStatus', 'Disable')