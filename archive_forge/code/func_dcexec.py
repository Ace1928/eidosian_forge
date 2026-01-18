import itertools
import logging
import os
import re
import subprocess
import time
import netaddr
def dcexec(self, cmd, capture=True, retry=False):
    if retry:
        return self.cmd.sudo(cmd, capture=capture, try_times=3, interval=1)
    else:
        return self.cmd.sudo(cmd, capture=capture)