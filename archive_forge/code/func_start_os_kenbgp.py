import logging
import os
import time
from . import docker_base as base
def start_os_kenbgp(self, check_running=True, retry=False):
    if check_running:
        if self.is_running_os_ken():
            return True
    result = False
    if retry:
        try_times = 3
    else:
        try_times = 1
    cmd = 'osken-manager --verbose '
    cmd += '--config-file %s ' % self.SHARED_OSKEN_CONF
    cmd += '--bgp-app-config-file %s ' % self.SHARED_BGP_CONF
    cmd += 'os_ken.services.protocols.bgp.application'
    for _ in range(try_times):
        self.exec_on_ctn(cmd, detach=True)
        if self.is_running_os_ken():
            result = True
            break
        time.sleep(1)
    return result