import itertools
import logging
import os
import re
import subprocess
import time
import netaddr
def get_bridges_dc(self):
    out = self.execute('docker network ls', sudo=True, retry=True)
    bridges = []
    for line in out.splitlines()[1:]:
        bridges.append(line.split()[1])
    return bridges