import logging
import socket
from subprocess import PIPE
from subprocess import Popen
import sys
import time
import traceback
import requests
from saml2test.check import CRITICAL
def stop_script_by_pid(pid):
    import os
    import signal
    os.kill(pid, signal.SIGKILL)