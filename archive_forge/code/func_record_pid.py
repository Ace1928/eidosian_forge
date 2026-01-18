import atexit
import errno
import logging
import os
import re
import subprocess
import sys
import threading
import time
import traceback
from paste.deploy import loadapp, loadserver
from paste.script.command import Command, BadCommand
def record_pid(self, pid_file):
    pid = os.getpid()
    if self.verbose > 1:
        print('Writing PID %s to %s' % (pid, pid_file))
    f = open(pid_file, 'w')
    f.write(str(pid))
    f.close()
    atexit.register(_remove_pid_file, pid, pid_file, self.verbose)