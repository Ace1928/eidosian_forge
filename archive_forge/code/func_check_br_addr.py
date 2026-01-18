import itertools
import logging
import os
import re
import subprocess
import time
import netaddr
def check_br_addr(self, br):
    ips = {}
    cmd = 'ip a show dev %s' % br
    for line in self.execute(cmd, sudo=True).split('\n'):
        if line.strip().startswith('inet '):
            elems = [e.strip() for e in line.strip().split(' ')]
            ips[4] = elems[1]
        elif line.strip().startswith('inet6 '):
            elems = [e.strip() for e in line.strip().split(' ')]
            ips[6] = elems[1]
    return ips