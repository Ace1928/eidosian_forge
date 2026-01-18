import itertools
import logging
import os
import re
import subprocess
import time
import netaddr
def sudo(self, cmd, capture=True, try_times=1, interval=1):
    cmd = 'sudo %s' % cmd
    return self.execute(cmd, capture=capture, try_times=try_times, interval=interval)