import sys, re, curl, exceptions
from the command line first, then standard input.
def do_dhcp_dns(self, line):
    index, address = line.split()
    if index in ('1', '2', '3'):
        self.session.set_DHCP_DNS_server(eval(index), address)
    else:
        print_stderr('linksys: server index out of bounds.')
    return 0