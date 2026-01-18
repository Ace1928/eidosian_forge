import sys, re, curl, exceptions
from the command line first, then standard input.
def do_upnp(self, line):
    self.flag_command(self.session.set_UPnP, line)
    return 0