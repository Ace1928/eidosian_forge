import errno
import os
import shlex
import signal
import sys
from collections import OrderedDict, UserList, defaultdict
from functools import partial
from subprocess import Popen
from time import sleep
from kombu.utils.encoding import from_utf8
from kombu.utils.objects import cached_property
from celery.platforms import IS_WINDOWS, Pidfile, signal_name
from celery.utils.nodenames import gethostname, host_format, node_format, nodesplit
from celery.utils.saferepr import saferepr
def handle_process_exit(self, retcode, on_signalled=None, on_failure=None):
    if retcode < 0:
        maybe_call(on_signalled, self, -retcode)
        return -retcode
    elif retcode > 0:
        maybe_call(on_failure, self, retcode)
    return retcode