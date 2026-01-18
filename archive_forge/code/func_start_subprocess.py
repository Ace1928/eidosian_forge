import configparser
import logging
import logging.handlers
import os
import signal
import sys
from oslo_rootwrap import filters
from oslo_rootwrap import subprocess
def start_subprocess(filter_list, userargs, exec_dirs=[], log=False, **kwargs):
    filtermatch = match_filter(filter_list, userargs, exec_dirs)
    command = filtermatch.get_command(userargs, exec_dirs)
    if log:
        logging.info('(%s > %s) Executing %s (filter match = %s)' % (_getlogin(), pwd.getpwuid(os.getuid())[0], command, filtermatch.name))

    def preexec():
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
        filtermatch.preexec()
    obj = subprocess.Popen(command, preexec_fn=preexec, env=filtermatch.get_environment(userargs), **kwargs)
    return obj