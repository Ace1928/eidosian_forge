import itertools
import logging
import os
import re
import subprocess
import time
import netaddr
def get_containers(self, allctn=False):
    cmd = 'docker ps --no-trunc=true'
    if allctn:
        cmd += ' --all=true'
    out = self.dcexec(cmd, retry=True)
    containers = []
    for line in out.splitlines()[1:]:
        containers.append(line.split()[-1])
    return containers