import logging
import socket
from subprocess import PIPE
from subprocess import Popen
import sys
import time
import traceback
import requests
from saml2test.check import CRITICAL
def stop_script_by_name(name):
    import os
    import signal
    import subprocess
    p = subprocess.Popen(['ps', '-A'], stdout=subprocess.PIPE)
    out, err = p.communicate()
    for line in out.splitlines():
        if name in line:
            pid = int(line.split(None, 1)[0])
            os.kill(pid, signal.SIGKILL)