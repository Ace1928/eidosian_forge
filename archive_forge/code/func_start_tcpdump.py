import itertools
import logging
import os
import re
import subprocess
import time
import netaddr
def start_tcpdump(self, interface=None, filename=None):
    if not interface:
        interface = 'eth0'
    if not filename:
        filename = '{0}/{1}.dump'.format(self.shared_volumes[0][1], interface)
    self.exec_on_ctn('tcpdump -i {0} -w {1}'.format(interface, filename), detach=True)