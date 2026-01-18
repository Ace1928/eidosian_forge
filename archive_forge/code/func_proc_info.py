import enum
import errno
import multiprocessing
import os
import stat
import time
import traceback
import psutil
from psutil import AIX
from psutil import BSD
from psutil import FREEBSD
from psutil import LINUX
from psutil import MACOS
from psutil import NETBSD
from psutil import OPENBSD
from psutil import OSX
from psutil import POSIX
from psutil import WINDOWS
from psutil._compat import PY3
from psutil._compat import long
from psutil._compat import unicode
from psutil.tests import CI_TESTING
from psutil.tests import VALID_PROC_STATUSES
from psutil.tests import PsutilTestCase
from psutil.tests import check_connection_ntuple
from psutil.tests import create_sockets
from psutil.tests import is_namedtuple
from psutil.tests import is_win_secure_system_proc
from psutil.tests import process_namespace
from psutil.tests import serialrun
def proc_info(pid):
    tcase = PsutilTestCase()

    def check_exception(exc, proc, name, ppid):
        tcase.assertEqual(exc.pid, pid)
        if exc.name is not None:
            tcase.assertEqual(exc.name, name)
        if isinstance(exc, psutil.ZombieProcess):
            tcase.assertProcessZombie(proc)
            if exc.ppid is not None:
                tcase.assertGreaterEqual(exc.ppid, 0)
                tcase.assertEqual(exc.ppid, ppid)
        elif isinstance(exc, psutil.NoSuchProcess):
            tcase.assertProcessGone(proc)
        str(exc)
        repr(exc)

    def do_wait():
        if pid != 0:
            try:
                proc.wait(0)
            except psutil.Error as exc:
                check_exception(exc, proc, name, ppid)
    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        tcase.assertPidGone(pid)
        return {}
    try:
        d = proc.as_dict(['ppid', 'name'])
    except psutil.NoSuchProcess:
        tcase.assertProcessGone(proc)
    else:
        name, ppid = (d['name'], d['ppid'])
        info = {'pid': proc.pid}
        ns = process_namespace(proc)
        for fun, fun_name in ns.iter(ns.getters, clear_cache=False):
            try:
                info[fun_name] = fun()
            except psutil.Error as exc:
                check_exception(exc, proc, name, ppid)
                continue
        do_wait()
        return info