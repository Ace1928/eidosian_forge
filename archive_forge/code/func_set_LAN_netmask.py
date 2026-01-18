import sys, re, curl, exceptions
from the command line first, then standard input.
def set_LAN_netmask(self, ip):
    if not ip.startswith('255.255.255.'):
        raise ValueError
    lastquad = ip.split('.')[-1]
    if lastquad not in ('0', '128', '192', '240', '252'):
        raise ValueError
    self.actions.append('', 'netMask', lastquad)