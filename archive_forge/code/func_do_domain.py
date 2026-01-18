import sys, re, curl, exceptions
from the command line first, then standard input.
def do_domain(self, line):
    print('Usage: host <domainname>')
    self.session.set_domain_name(line)
    return 0