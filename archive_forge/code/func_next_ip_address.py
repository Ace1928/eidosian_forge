import itertools
import logging
import os
import re
import subprocess
import time
import netaddr
def next_ip_address(self):
    return '{0}/{1}'.format(next(self._ip_generator), self.subnet.prefixlen)